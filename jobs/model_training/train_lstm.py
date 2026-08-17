import os
import sys
import math
import logging

# Configure Logging FIRST
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ML_Training")

logger.info("Script started! Importing PyTorch...")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

logger.info("Importing PyArrow, NumPy, and PyIceberg...")
import pyarrow as pa
import numpy as np
from pyiceberg.catalog import load_catalog

logger.info("Importing MLflow...")
import mlflow
import mlflow.pytorch

logger.info("All heavy libraries imported successfully! Setting up hyperparameters...")

# ==========================================
# HYPERPARAMETERS (OPTIMIZED FOR PURE SPEED)
# ==========================================
SEQ_LEN = 72         # 3 days of context
PRED_LEN = 24        # 1 day prediction
BATCH_SIZE = 2048     # High throughput
EPOCHS = 1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  
FEATURE_DIM = 4      
HIDDEN_DIM = 64
DROPOUT = 0.2

# ==========================================
# DATASET DEFINITION
# ==========================================
class IcebergTimeSeriesDataset(Dataset):
    def __init__(self, table_name: str, seq_len: int, pred_len: int):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        logger.info(f"Connecting to Iceberg catalog to load {table_name}...")
        catalog = load_catalog(
            "default",
            **{
                "type": "rest",
                "uri": "http://nessie:19120/iceberg/main",
                "s3.endpoint": "http://minio:9000",
                "s3.access-key-id": os.environ["AWS_ACCESS_KEY_ID"],
                "s3.secret-access-key": os.environ["AWS_SECRET_ACCESS_KEY"],
                "s3.path-style-access": "true"
            }
        )
        
        table = catalog.load_table(("weather", "ml_features"))
        logger.info("Executing table scan and converting to Arrow...")
        arrow_table = table.scan(selected_fields=("features",)).to_arrow()
        
        col = arrow_table.column("features")
        logger.info("Converting column chunks to numpy array...")
        
        flattened_data = []
        for chunk in col.chunks:
            flattened_data.extend(chunk.to_pylist())
            
        logger.info("Constructing PyTorch tensor from extracted data...")
        self.data = torch.tensor(flattened_data, dtype=torch.float32)
        
        self.total_windows = len(self.data) - self.seq_len - self.pred_len + 1
        logger.info(f"Dataset ready. Generated {self.total_windows} sliding windows.")

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_len
        target_end = end_idx + self.pred_len
        
        x = self.data[start_idx : end_idx]
        y = self.data[end_idx : target_end]
        
        return x, y

# ==========================================
# ADVANCED MODEL ARCHITECTURE (CONV-LSTM)
# ==========================================
class ConvLSTMWeatherForecaster(nn.Module):
    """Conv1D + LSTM hybrid to smooth missing data and capture temporal sequence."""
    def __init__(self, input_dim, hidden_dim, output_dim, pred_len, dropout):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        # 1. Localized Feature Extractor (Replaces Linear Proj + Positional Encoding)
        # Kernel size 3 acts as a learned moving average over 3 hours
        self.feature_extractor = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1 # Padding=1 keeps the sequence length identical
        )
        
        # 2. Recurrent memory layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        
        # 3. Deep non-linear projection head with LayerNorm for stability
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, pred_len * output_dim)
        )

    def forward(self, x):
        # PyTorch Conv1d expects shape (Batch, Channels, Sequence)
        # Input 'x' is currently (Batch, Sequence, Channels)
        x = x.transpose(1, 2)
        
        # Extract features and smooth data
        x = self.feature_extractor(x)
        
        # Revert shape for the LSTM: (Batch, Sequence, Channels)
        x = x.transpose(1, 2)
        
        # Extract sequential states
        lstm_out, _ = self.lstm(x) 
        
        # Pool the temporal dimension by taking only the last state
        last_state = lstm_out[:, -1, :] 
        
        # Generate predictions
        out = self.fc_head(last_state)
        return out.view(-1, self.pred_len, self.output_dim)

# ==========================================
# TRAINING LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    logger.info(f"Initializing training on device: {device}")

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Weather_Forecaster_FastLSTM")

    # 1. Load Data
    full_dataset = IcebergTimeSeriesDataset("nessie.weather.ml_features", seq_len=SEQ_LEN, pred_len=PRED_LEN)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # 2. Initialize Lean Model
    model = ConvLSTMWeatherForecaster(
        input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, output_dim=FEATURE_DIM, 
        pred_len=PRED_LEN, dropout=DROPOUT
    ).to(device)
    
    MODEL_PATH = "weather_fast_lstm_best.pth"
    
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    scaler = torch.amp.GradScaler('cuda')

    # 3. MLflow Run Execution
    with mlflow.start_run():
        logger.info("Starting High-Speed Conv-LSTM training loop...")
        
        mlflow.log_params({
            "model_architecture": "Conv-LSTM",
            "seq_len": SEQ_LEN, "pred_len": PRED_LEN, "batch_size": BATCH_SIZE,
            "hidden_dim": HIDDEN_DIM, 
            "initial_lr": LEARNING_RATE, "optimizer": "AdamW", "loss": "SmoothL1Loss"
        })

        best_val_loss = float("inf")

        for epoch in range(EPOCHS):
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
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Train Batch {batch_idx} | Loss (SmoothL1): {loss.item():.4f}")

            train_loss = train_running_loss / len(train_loader)

            # --- VALIDATION PHASE ---
            model.eval()
            val_running_loss = 0.0
            
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device, non_blocking=True)
                    y_val = y_val.to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda'):
                        val_preds = model(x_val)
                        val_loss_batch = criterion(val_preds, y_val)
                        
                    val_running_loss += val_loss_batch.item()

            val_loss = val_running_loss / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.info(f"--- Epoch {epoch+1} Complete | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} ---")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "learning_rate": current_lr}, step=epoch)

            # Step the Cosine schedule every epoch
            scheduler.step()

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                logger.info(f"New best model saved! (Val Loss: {best_val_loss:.4f})")

        # 4. Log Best Model to MLflow
        logger.info("Loading best weights and logging full model artifact to MLflow...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        # Move model to CPU for a clean, device-agnostic save
        model.to("cpu")
        sample_input, _ = next(iter(val_loader))
        
        mlflow.pytorch.log_model(
            model, 
            name="weather_fast_lstm_model",  # Updated from artifact_path to name
            input_example=sample_input.numpy(),
            serialization_format="pickle"
        )
            
        logger.info("MLflow model logging completed.")

if __name__ == "__main__":
    main()