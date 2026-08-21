import os
import sys
import math
import logging
from typing import NamedTuple

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Model_Evaluation")

import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from baselines import NAIVE_FORECASTS
from data_loader import get_dataloaders
from promotion import promotion_verdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 72
PRED_LEN = 24
# Deliberately no autocast here, unlike training. The promotion threshold is 1% and
# the observed margins have been ~0.2%, which is inside fp16 noise - a benchmark that
# decides deployments has to be reproducible.

class Scores(NamedTuple):
    loss: float
    rmse: float
    mae: float
    window_mse: "object"   # np.ndarray, one mean squared error per test window


class _Running:
    """Per-predictor accumulators for one pass over the loader."""

    def __init__(self):
        self.loss = 0.0
        self.sq_error = 0.0
        self.abs_error = 0.0
        self.elements = 0
        self.window_mse = []


def score_predictors(predictors, dataloader, criterion):
    """Scores every predictor in a single pass, returning {name: Scores}.

    Champion, challenger and the naive baselines are all scored on identical windows,
    so walking the test block once per predictor multiplied the cost of this task by
    four for nothing - and the block is ~113k windows inside a 15-minute timeout.

    Callables rather than models, so the baselines, which have no forward pass, go
    through exactly the same code as the two networks.

    RMSE is the real thing - sqrt of the mean squared error - not sqrt of the
    SmoothL1 objective, and both errors are accumulated over elements so a short
    final batch does not get the same weight as a full one.
    """
    if not predictors:
        return {}

    running = {name: _Running() for name in predictors}

    with torch.no_grad():
        for x_val, y_val in dataloader:
            x_val = x_val.to(device, non_blocking=True)
            y_val = y_val.to(device, non_blocking=True)

            for name, predict in predictors.items():
                acc = running[name]
                preds = predict(x_val)
                acc.loss += criterion(preds, y_val).item()

                diff = preds.float() - y_val.float()
                acc.sq_error += torch.sum(diff * diff).item()
                acc.abs_error += torch.sum(diff.abs()).item()
                acc.elements += diff.numel()
                # One error per window, kept for the paired promotion test. The loader
                # is unshuffled, so these stay aligned across predictors.
                acc.window_mse.append((diff * diff).mean(dim=(1, 2)).cpu())

    batches = len(dataloader)
    return {
        name: Scores(
            loss=acc.loss / batches,
            rmse=math.sqrt(acc.sq_error / acc.elements),
            mae=acc.abs_error / acc.elements,
            window_mse=torch.cat(acc.window_mse).numpy(),
        )
        for name, acc in running.items()
    }

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
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=128,
        is_incremental=False
    )
    criterion = nn.SmoothL1Loss()

    # 5. Load both models for head-to-head comparison
    logger.info(f"Loading Challenger (v{challenger_version}) from MLflow...")
    challenger_model = mlflow.pytorch.load_model(f"models:/{model_name}/{challenger_version}", map_location=device)
    
    logger.info(f"Loading Champion (v{champion_version}) from MLflow...")
    champion_model = mlflow.pytorch.load_model(f"models:/{model_name}/{champion_version}", map_location=device)

    # 6. Execute Benchmark Arena. One pass over the test block scores all four:
    # both models, plus two forecasts that were never trained. The baselines gate
    # nothing - they are here so the numbers above can be read at all. A champion
    # that cannot beat seasonal-naive is not a forecaster.
    challenger_model.eval()
    champion_model.eval()
    scores = score_predictors(
        {
            "champion": champion_model,
            "challenger": challenger_model,
            **{name: (lambda x, f=fn: f(x, PRED_LEN)) for name, fn in NAIVE_FORECASTS.items()},
        },
        test_loader,
        criterion,
    )
    champion = scores["champion"]
    challenger = scores["challenger"]
    naive = {name: scores[name] for name in NAIVE_FORECASTS}

    logger.info(f"--- BENCHMARK RESULTS ---")
    logger.info(f"Champion   (v{champion_version}):   Loss={champion.loss:.4f} | RMSE={champion.rmse:.4f} | MAE={champion.mae:.4f}")
    logger.info(f"Challenger (v{challenger_version}): Loss={challenger.loss:.4f} | RMSE={challenger.rmse:.4f} | MAE={challenger.mae:.4f}")
    for name, scores in naive.items():
        skill = 1.0 - (champion.rmse / scores.rmse) if scores.rmse else float("nan")
        logger.info(
            f"Baseline   ({name}): RMSE={scores.rmse:.4f} | MAE={scores.mae:.4f} "
            f"| champion skill vs it: {skill:+.1%}"
        )

    # 7. Log evaluation metrics to MLflow
    with mlflow.start_run(run_name=f"Eval_v{challenger_version}_vs_v{champion_version}"):
        mlflow.log_params({
            "model_name": model_name,
            "champion_version": champion_version,
            "challenger_version": challenger_version,
        })
        metrics = {
            "champion_rmse": champion.rmse,
            "champion_mae": champion.mae,
            "challenger_rmse": challenger.rmse,
            "challenger_mae": challenger.mae,
            "rmse_diff": challenger.rmse - champion.rmse,
        }
        for name, scores in naive.items():
            metrics[f"baseline_{name}_rmse"] = scores.rmse
            metrics[f"baseline_{name}_mae"] = scores.mae
            # Positive means the model beats the baseline; negative means it does not.
            if scores.rmse:
                metrics[f"champion_skill_vs_{name}"] = 1.0 - (champion.rmse / scores.rmse)
                metrics[f"challenger_skill_vs_{name}"] = 1.0 - (challenger.rmse / scores.rmse)

        # 8. Promotion decision: a floor worth deploying for, and a paired test on
        # disjoint windows. See promotion.py for why a fixed percentage was wrong.
        verdict = promotion_verdict(champion.window_mse, challenger.window_mse,
                                    stride=SEQ_LEN + PRED_LEN)
        metrics["promotion_relative_improvement"] = verdict.relative_improvement
        metrics["promotion_p_value"] = verdict.p_value
        metrics["promotion_disjoint_windows"] = float(verdict.samples)
        mlflow.log_metrics(metrics)

        if verdict.promote:
            logger.info(f"PROMOTION SUCCESSFUL! Challenger v{challenger_version} beat Champion v{champion_version}: {verdict.reason}.")
            client.set_registered_model_alias(name=model_name, alias="champion", version=challenger_version)
            client.set_registered_model_alias(name=model_name, alias="previous_champion", version=champion_version)
            # A re-evaluated version can carry a stale rejection from an earlier run;
            # leaving it makes the registry claim the same version is both champion
            # and rejected.
            try:
                stale = client.get_model_version_by_alias(name=model_name, alias="challenger_rejected")
                if str(stale.version) == challenger_version:
                    client.delete_registered_model_alias(name=model_name, alias="challenger_rejected")
            except Exception:
                pass
            mlflow.log_param("promotion_decision", "PROMOTED")
        else:
            logger.warning(f"PROMOTION REJECTED. Challenger v{challenger_version} stays behind Champion v{champion_version}: {verdict.reason}.")
            client.set_registered_model_alias(name=model_name, alias="challenger_rejected", version=challenger_version)
            mlflow.log_param("promotion_decision", "REJECTED")

if __name__ == "__main__":
    target_model_name = os.getenv("MODEL_REGISTRY_NAME")
    if not target_model_name:
        logger.error("Environment variable 'MODEL_REGISTRY_NAME' must be set.")
        sys.exit(1)
        
    evaluate_and_promote(model_name=target_model_name)