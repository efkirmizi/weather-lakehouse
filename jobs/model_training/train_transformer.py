import os
import sys
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ML_Training")

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import DEFAULT_MAX_FEATURE_AGE_HOURS, get_dataloaders
from lakehouse import INPUT_CHANNELS, OUTPUT_CHANNELS, PRED_LEN, SEQ_LEN
from models import TimeSeriesTransformer
from trainer import get_champion_weights, resolve_epochs, train_and_register_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# CONFIGURATION
CONFIG = {
    "model_registry_name": "Weather_Forecaster_Transformer",
    "table_name": "weather.ml_features",
    "seq_len": SEQ_LEN, "pred_len": PRED_LEN, "batch_size": 128,
    "input_dim": INPUT_CHANNELS, "output_dim": OUTPUT_CHANNELS,
    "d_model": 32, "n_heads": 2,
    "num_layers": 4, "dim_feedforward": 128, "dropout": 0.1,
    "patience": 4, "weight_decay": 1e-4
}

def main():
    logger.info(f"Initializing Transformer training on device: {device}")
    
    mode = os.getenv("TRAINING_MODE", "SCRATCH")
    is_incremental = (mode == "INCREMENTAL")

    prev_weights = None
    if is_incremental:
        prev_weights = get_champion_weights(CONFIG["model_registry_name"], device)
        if prev_weights is None:
            is_incremental = False

    epochs = resolve_epochs(is_incremental, scratch=10, incremental=2)
    lr = 1e-4 if is_incremental else 0.001
    # Warmup has to leave at least one epoch for the cosine phase, otherwise
    # CosineAnnealingLR gets a non-positive T_max and the schedule degenerates.
    warmup_epochs = min(1 if is_incremental else 2, max(epochs - 1, 0))

    # The test split is deliberately ignored here: it belongs to the promotion gate,
    # and a model early-stopped on it would be benchmarked on its own tuning data.
    train_loader, val_loader, _ = get_dataloaders(
        CONFIG["table_name"], CONFIG["seq_len"], CONFIG["pred_len"], CONFIG["batch_size"],
        is_incremental, max_age_hours=DEFAULT_MAX_FEATURE_AGE_HOURS
    )

    model = TimeSeriesTransformer(
        CONFIG["input_dim"], CONFIG["d_model"], CONFIG["n_heads"], CONFIG["num_layers"],
        CONFIG["dim_feedforward"], CONFIG["output_dim"], CONFIG["pred_len"],
        CONFIG["dropout"]
    ).to(device)

    if prev_weights:
        model.load_state_dict(prev_weights)
        logger.info("Successfully warm-started model with previous weights.")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    if warmup_epochs > 0:
        scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
    else:
        # A single-epoch smoke run has no room for a warmup phase.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    hyperparams = {
        **CONFIG,
        # See train_lstm.py: the effective mode, not the requested one.
        "training_mode": "INCREMENTAL" if is_incremental else "SCRATCH",
        "requested_mode": mode,
        "warm_started": prev_weights is not None,
        "model_type": "Transformer", "epochs": epochs, "initial_lr": lr, "optimizer": "AdamW",
    }

    train_and_register_model(
        model=model, 
        model_registry_name=CONFIG["model_registry_name"], 
        experiment_name=CONFIG["model_registry_name"], 
        train_loader=train_loader, 
        val_loader=val_loader,
        optimizer=optimizer, 
        scheduler=scheduler, 
        criterion=nn.SmoothL1Loss(), 
        epochs=epochs, 
        patience=CONFIG["patience"], 
        device=device, 
        hyperparams_dict=hyperparams
    )

if __name__ == "__main__":
    main()