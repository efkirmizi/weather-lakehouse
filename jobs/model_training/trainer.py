import os
import math
import torch
import torch.quantization
import logging
import mlflow
import mlflow.pytorch
import mlflow.onnx
import onnx
from mlflow.tracking import MlflowClient

from lakehouse import ONNX_ARTIFACT_NAME, ONNX_OPSET

logger = logging.getLogger("ML_Training")

MLFLOW_TRACKING_URI = "http://mlflow:5000"
S3_ENDPOINT_URL = "http://minio:9000"


def _configure_mlflow() -> None:
    """Point this process at the tracking server and at MinIO.

    Every entry point that touches the registry has to call this first. A bare
    MlflowClient() resolves to the container's own ./mlruns, which in an ephemeral job
    container is always empty: the lookup then reports "no model registered" however
    healthy the server is, and artifact downloads go to real AWS S3 instead of MinIO.
    That is exactly how warm-starting failed silently - the branch operator queries the
    REST API directly and saw the model, the training container did not.
    """
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT_URL
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def resolve_epochs(is_incremental: bool, scratch: int, incremental: int) -> int:
    """Epoch budget for this run, overridable per DAG run via TRAINING_EPOCHS.

    The DAG forwards dag_run.conf['TRAINING_EPOCHS'], so a pipeline change can be
    exercised end to end in minutes without weakening the production defaults.
    """
    default = incremental if is_incremental else scratch
    override = os.getenv("TRAINING_EPOCHS", "").strip()
    if not override:
        return default

    try:
        value = int(override)
    except ValueError:
        logger.warning(f"Ignoring non-integer TRAINING_EPOCHS={override!r}; using {default}.")
        return default

    if value < 1:
        logger.warning(f"Ignoring TRAINING_EPOCHS={value}; using {default}.")
        return default

    logger.warning(f"TRAINING_EPOCHS override active: {value} epoch(s) instead of {default}.")
    return value


def get_champion_weights(registry_name, device):
    """Weights to warm-start an incremental run from, or None to train from scratch.

    Deliberately the '@champion' alias rather than the newest registered version. A
    challenger that fails the promotion gate is still registered, so resuming from
    max(version) builds on a model the gate has already rejected - and compounds it
    every week, since the next rejected challenger becomes the next warm start. The
    champion is the only version the pipeline has evidence for.
    """
    _configure_mlflow()
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(name=registry_name, alias="champion")
    except Exception as e:
        logger.info(f"No '@champion' alias on '{registry_name}' ({e}). Training from scratch.")
        return None

    logger.info(f"Warm-starting from '{registry_name}@champion' (version {champion.version}).")
    try:
        model = mlflow.pytorch.load_model(
            f"models:/{registry_name}/{champion.version}", map_location=device
        )
        return model.state_dict()
    except Exception as e:
        logger.info(f"Could not load the champion's weights ({e}). Training from scratch.")
        return None


def warm_start(model, weights) -> bool:
    """Loads champion weights into a freshly built model. True if they fitted.

    The caller must treat False as "this run is a scratch run" - not just for the
    weights, but for the epoch budget, the learning rate and the data window, all of
    which get_dataloaders and resolve_epochs pick from the same flag. Two incremental
    epochs at 1e-4 on the recent-window split is a fine way to adapt a champion and a
    terrible way to train a randomly initialised model.

    The mismatch case is not hypothetical: any hyperparameter that changes a tensor
    shape - a wider d_model, another layer, a different input width - leaves the
    champion's state dict incompatible with the architecture just built. strict
    load_state_dict raises there, so the weekly INCREMENTAL run would fail as a DAG
    error the first Sunday after any such change, when the obviously right response is
    the one get_champion_weights already takes when it cannot fetch the champion at
    all: there is nothing to resume from, so start over.
    """
    if not weights:
        return False
    try:
        model.load_state_dict(weights)
    except RuntimeError as e:
        logger.info(
            f"The champion's weights do not fit the current architecture ({e}). "
            "Training from scratch."
        )
        return False
    logger.info("Warm-started from the champion's weights.")
    return True


def _export_onnx(model, sample_input, onnx_path):
    """Exports the trained float model to ONNX with a dynamic batch dimension."""
    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        input_names=["input"],
        output_names=["forecast"],
        dynamic_axes={"input": {0: "batch_size"}, "forecast": {0: "batch_size"}},
        opset_version=ONNX_OPSET,
    )
    return onnx.load(onnx_path)


def train_and_register_model(
    model, model_registry_name, experiment_name,
    train_loader, val_loader, optimizer, scheduler, criterion,
    epochs, patience, device, hyperparams_dict
):
    _configure_mlflow()
    mlflow.set_experiment(experiment_name)

    temp_model_path = f"{model_registry_name}_best_temp.pth"
    # Mixed precision is a CUDA feature; on CPU the same code path has to run unscaled.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    with mlflow.start_run():
        logger.info(f"Starting Training Engine for {model_registry_name} (AMP={'on' if use_amp else 'off'})...")
        mlflow.log_params(hyperparams_dict)

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # --- TRAIN PHASE ---
            model.train()
            train_running_loss = 0.0

            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device.type, enabled=use_amp):
                    predictions = model(x_batch)
                    loss = criterion(predictions, y_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                train_running_loss += loss.item()

            train_loss = train_running_loss / len(train_loader)

            # --- VALIDATION PHASE ---
            model.eval()
            val_running_loss = 0.0
            # Accumulated over elements, not batches: sqrt of the SmoothL1 objective
            # is not an RMSE, and a mean of batch means mis-weights a short last batch.
            sq_error_sum = 0.0
            abs_error_sum = 0.0
            element_count = 0

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device, non_blocking=True)
                    y_val = y_val.to(device, non_blocking=True)

                    with torch.amp.autocast(device.type, enabled=use_amp):
                        val_preds = model(x_val)
                        val_loss_batch = criterion(val_preds, y_val)

                    val_running_loss += val_loss_batch.item()
                    diff = val_preds.float() - y_val.float()
                    sq_error_sum += torch.sum(diff * diff).item()
                    abs_error_sum += torch.sum(diff.abs()).item()
                    element_count += diff.numel()

            val_loss = val_running_loss / len(val_loader)
            val_rmse = math.sqrt(sq_error_sum / element_count)
            val_mae = abs_error_sum / element_count
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(f"--- Epoch {epoch+1} Complete | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} ---")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "learning_rate": current_lr
            }, step=epoch)

            scheduler.step()

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), temp_model_path)
                logger.info(f"New best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.warning("Early stopping triggered to prevent overfitting.")
                break

        # --- LOG AND REGISTER MODEL ---
        logger.info("Loading best weights for export and MLflow registration...")
        model.load_state_dict(torch.load(temp_model_path, map_location=device))
        model.to("cpu")
        model.eval()

        # One row is enough for the signature and for tracing the ONNX graph; the
        # batch dimension is exported as dynamic anyway.
        sample_batch, _ = next(iter(val_loader))
        sample_input = sample_batch[:1]

        # 1. The registry entry stays a float PyTorch model. Warm-starting the next
        #    incremental run and the champion/challenger benchmark both load it back
        #    as a live nn.Module, which a quantized state_dict cannot satisfy.
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=model_registry_name,
            input_example=sample_input.numpy(),
            serialization_format="pickle"
        )

        # 2. The serving graph. batch_inference.py runs this through ONNX Runtime.
        onnx_path = f"{model_registry_name}_serving.onnx"
        logger.info("Exporting float model to ONNX for the serving path...")
        onnx_model = _export_onnx(model, sample_input, onnx_path)
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            name=ONNX_ARTIFACT_NAME,
            input_example=sample_input.numpy()
        )

        # 3. Dynamic INT8 quantization, kept as a plain artifact for size comparison.
        #    It is deliberately not the registered model: dynamic quant is CPU-only
        #    and its state_dict keys do not map back onto the float architecture.
        logger.info("Applying dynamic INT8 quantization to linear and recurrent layers...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8
        )
        quant_path = f"{temp_model_path}.quant"
        torch.save(quantized_model.state_dict(), quant_path)

        orig_size = os.path.getsize(temp_model_path) / 1e6
        quant_size = os.path.getsize(quant_path) / 1e6
        logger.info(f"Model compressed from {orig_size:.2f} MB to {quant_size:.2f} MB")
        mlflow.log_metrics({"model_size_mb": orig_size, "quantized_size_mb": quant_size})
        mlflow.log_artifact(quant_path, artifact_path="quantized")

        logger.info("MLflow model logging & registration completed.")
