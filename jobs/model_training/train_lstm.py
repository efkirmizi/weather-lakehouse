import os
import sys
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ML_Training")

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import DEFAULT_MAX_FEATURE_AGE_HOURS, get_dataloaders
from models import ConvLSTMWeatherForecaster
from trainer import get_latest_model_weights, resolve_epochs, train_and_register_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# CONFIGURATION
CONFIG = {
    "model_registry_name": "Weather_Forecaster_FastLSTM",
    "table_name": "weather.ml_features",
    "seq_len": 72, "pred_len": 24, "batch_size": 128,
    "feature_dim": 4, "hidden_dim": 64, "dropout": 0.2,
    "patience": 4, "weight_decay": 1e-4
}

def main():
    logger.info(f"Initializing LSTM training on device: {device}")
    
    mode = os.getenv("TRAINING_MODE", "SCRATCH")
    is_incremental = (mode == "INCREMENTAL")

    prev_weights = None
    if is_incremental:
        prev_weights = get_latest_model_weights(CONFIG["model_registry_name"], device)
        if prev_weights is None:
            is_incremental = False

    epochs = resolve_epochs(is_incremental, scratch=10, incremental=2)
    lr = 1e-4 if is_incremental else 0.001

    # The test split is deliberately ignored here: it belongs to the promotion gate,
    # and a model early-stopped on it would be benchmarked on its own tuning data.
    train_loader, val_loader, _ = get_dataloaders(
        CONFIG["table_name"], CONFIG["seq_len"], CONFIG["pred_len"], CONFIG["batch_size"],
        is_incremental, max_age_hours=DEFAULT_MAX_FEATURE_AGE_HOURS
    )

    model = ConvLSTMWeatherForecaster(
        CONFIG["feature_dim"], CONFIG["hidden_dim"], CONFIG["feature_dim"], 
        CONFIG["pred_len"], CONFIG["dropout"]
    ).to(device)

    if prev_weights:
        model.load_state_dict(prev_weights)
        logger.info("Successfully warm-started Conv-LSTM model with previous weights.")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    hyperparams = {**CONFIG, "training_mode": mode, "model_architecture": "Conv-LSTM", "epochs": epochs, "initial_lr": lr, "optimizer": "AdamW"}

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