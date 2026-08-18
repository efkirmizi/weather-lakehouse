import os
import math
import torch
import logging
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

logger = logging.getLogger("ML_Training")

def get_latest_model_weights(registry_name, device):
    """Fetches the latest registered model from MLflow, or returns None if starting from scratch."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{registry_name}'")
        if not versions:
            logger.info("No existing model found in MLflow Registry. Training from scratch (Tabula Rasa).")
            return None
            
        latest_version = max(versions, key=lambda v: int(v.version))
        logger.info(f"Found existing model: Version {latest_version.version}. Preparing for Incremental Learning.")
        
        model_uri = f"models:/{registry_name}/{latest_version.version}"
        prev_model = mlflow.pytorch.load_model(model_uri, map_location=device)
        return prev_model.state_dict()
        
    except Exception as e:
        logger.info(f"Could not load previous model from registry ({e}). Training from scratch.")
        return None


def train_and_register_model(
    model, model_registry_name, experiment_name,
    train_loader, val_loader, optimizer, scheduler, criterion,
    epochs, patience, device, hyperparams_dict
):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(experiment_name)

    temp_model_path = f"{model_registry_name}_best_temp.pth"
    scaler = torch.amp.GradScaler('cuda')

    with mlflow.start_run():
        logger.info(f"Starting Training Engine for {model_registry_name}...")
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
                
                with torch.amp.autocast('cuda'):
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
            val_mae_sum = 0.0
            
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device, non_blocking=True)
                    y_val = y_val.to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda'):
                        val_preds = model(x_val)
                        val_loss_batch = criterion(val_preds, y_val)
                        
                    val_running_loss += val_loss_batch.item()
                    val_mae_sum += torch.abs(val_preds - y_val).mean().item()

            val_loss = val_running_loss / len(val_loader)
            val_rmse = math.sqrt(val_loss)
            val_mae = val_mae_sum / len(val_loader)
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
        logger.info("Loading best weights and registering model to MLflow Repository...")
        model.load_state_dict(torch.load(temp_model_path, map_location=device))
        
        model.to("cpu")
        sample_input, _ = next(iter(val_loader))
        
        mlflow.pytorch.log_model(
            pytorch_model=model, 
            name="model",
            registered_model_name=model_registry_name,
            input_example=sample_input.numpy(),
            serialization_format="pickle"
        )
            
        logger.info("MLflow model logging & registration completed.")