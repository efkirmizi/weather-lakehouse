import os
import sys
import math
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Model_Evaluation")

import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from data_loader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Deliberately no autocast here, unlike training. The promotion threshold is 1% and
# the observed margins have been ~0.2%, which is inside fp16 noise - a benchmark that
# decides deployments has to be reproducible.

def compute_metrics(model, dataloader, criterion):
    """Evaluates a model over a DataLoader and computes Loss, RMSE and MAE.

    RMSE is the real thing - sqrt of the mean squared error - not sqrt of the
    SmoothL1 objective, and both errors are accumulated over elements so a short
    final batch does not get the same weight as a full one.
    """
    model.eval()
    total_loss = 0.0
    sq_error_sum = 0.0
    abs_error_sum = 0.0
    element_count = 0

    with torch.no_grad():
        for x_val, y_val in dataloader:
            x_val = x_val.to(device, non_blocking=True)
            y_val = y_val.to(device, non_blocking=True)

            preds = model(x_val)
            total_loss += criterion(preds, y_val).item()

            diff = preds.float() - y_val.float()
            sq_error_sum += torch.sum(diff * diff).item()
            abs_error_sum += torch.sum(diff.abs()).item()
            element_count += diff.numel()

    avg_loss = total_loss / len(dataloader)
    rmse = math.sqrt(sq_error_sum / element_count)
    mae = abs_error_sum / element_count
    return avg_loss, rmse, mae

def evaluate_and_promote(model_name: str, table_name: str = "weather.ml_features"):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(f"{model_name}_Evaluations")
    client = MlflowClient()

    # 1. Fetch the latest registered version (The Challenger)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logger.error(f"No model versions found in registry for '{model_name}'.")
        sys.exit(1)
        
    challenger_ver_obj = max(versions, key=lambda v: int(v.version))
    challenger_version = str(challenger_ver_obj.version)
    logger.info(f"Identified Challenger model version: v{challenger_version}")

    # 2. Check if a Champion exists
    champion_version = None
    try:
        champion_model_obj = client.get_model_version_by_alias(name=model_name, alias="champion")
        champion_version = str(champion_model_obj.version)
        logger.info(f"Current Champion model version: v{champion_version}")
    except Exception:
        logger.info("No Champion currently crowned. First registered model will be promoted automatically.")

    # 3. First-time deployment needs no benchmark - decide before loading 750k windows.
    if champion_version is None or champion_version == challenger_version:
        logger.info(f"Promoting v{challenger_version} to '@champion' (Initial Baseline)...")
        client.set_registered_model_alias(name=model_name, alias="champion", version=challenger_version)
        logger.info("Model successfully crowned Champion!")
        return

    # 4. Load the held-out test split. Training early-stops on the validation split,
    # so scoring the challenger there would judge it on its own tuning data. The test
    # block is chronological and seeded, so both models see identical windows.
    *_, test_loader = get_dataloaders(
        table_name=table_name,
        seq_len=72,
        pred_len=24,
        batch_size=128,
        is_incremental=False
    )
    criterion = nn.SmoothL1Loss()

    # 5. Load both models for head-to-head comparison
    logger.info(f"Loading Challenger (v{challenger_version}) from MLflow...")
    challenger_model = mlflow.pytorch.load_model(f"models:/{model_name}/{challenger_version}", map_location=device)
    
    logger.info(f"Loading Champion (v{champion_version}) from MLflow...")
    champion_model = mlflow.pytorch.load_model(f"models:/{model_name}/{champion_version}", map_location=device)

    # 6. Execute Benchmark Arena
    challenger_loss, challenger_rmse, challenger_mae = compute_metrics(challenger_model, test_loader, criterion)
    champion_loss, champion_rmse, champion_mae = compute_metrics(champion_model, test_loader, criterion)

    logger.info(f"--- BENCHMARK RESULTS ---")
    logger.info(f"Champion   (v{champion_version}):   Loss={champion_loss:.4f} | RMSE={champion_rmse:.4f} | MAE={champion_mae:.4f}")
    logger.info(f"Challenger (v{challenger_version}): Loss={challenger_loss:.4f} | RMSE={challenger_rmse:.4f} | MAE={challenger_mae:.4f}")

    # 7. Log evaluation metrics to MLflow
    with mlflow.start_run(run_name=f"Eval_v{challenger_version}_vs_v{champion_version}"):
        mlflow.log_params({
            "model_name": model_name,
            "champion_version": champion_version,
            "challenger_version": challenger_version,
        })
        mlflow.log_metrics({
            "champion_rmse": champion_rmse,
            "champion_mae": champion_mae,
            "challenger_rmse": challenger_rmse,
            "challenger_mae": challenger_mae,
            "rmse_diff": challenger_rmse - champion_rmse
        })

        # 8. Promotion Threshold Decision (1% improvement required or equal/better)
        PROMOTION_THRESHOLD = 0.99  # Challenger RMSE must be <= 99% of Champion RMSE
        if challenger_rmse <= (champion_rmse * PROMOTION_THRESHOLD):
            logger.info(f"PROMOTION SUCCESSFUL! Challenger v{challenger_version} beat Champion v{champion_version}.")
            client.set_registered_model_alias(name=model_name, alias="champion", version=challenger_version)
            client.set_registered_model_alias(name=model_name, alias="previous_champion", version=champion_version)
            mlflow.log_param("promotion_decision", "PROMOTED")
        else:
            logger.warning(f"PROMOTION REJECTED. Challenger v{challenger_version} failed to outperform Champion v{champion_version}.")
            client.set_registered_model_alias(name=model_name, alias="challenger_rejected", version=challenger_version)
            mlflow.log_param("promotion_decision", "REJECTED")

if __name__ == "__main__":
    target_model_name = os.getenv("MODEL_REGISTRY_NAME")
    if not target_model_name:
        logger.error("Environment variable 'MODEL_REGISTRY_NAME' must be set.")
        sys.exit(1)
        
    evaluate_and_promote(model_name=target_model_name)