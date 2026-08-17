import os
import sys
import math
import logging

"""
# FORCE SINGLE-THREADED C++ EXTENSIONS TO PREVENT IMPORT DEADLOCKS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
"""

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
# HYPERPARAMETERS (OPTIMIZED)
# ==========================================
SEQ_LEN = 72         
PRED_LEN = 24
BATCH_SIZE = 2048     
EPOCHS = 1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
FEATURE_DIM = 4
D_MODEL = 32
N_HEAD = 2
NUM_LAYERS = 4
DIM_FEEDFORWARD = 128 
DROPOUT = 0.1
PATIENCE = 4         

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
# ADVANCED ARCHITECTURE: TRANSFORMER
# ==========================================
class PositionalEncoding(nn.Module):
    """Injects mathematical chronological awareness into the sequence."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(nn.Module):
    """Transformer Encoder for multivariate time-series forecasting."""
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, pred_len, dropout):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, pred_len * output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        pooled_output = encoded.mean(dim=1) 
        out = self.output_projection(pooled_output)
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
    mlflow.set_experiment("Weather_Forecaster_Transformer")

    full_dataset = IcebergTimeSeriesDataset("nessie.weather.ml_features", seq_len=SEQ_LEN, pred_len=PRED_LEN)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    model = TimeSeriesTransformer(
        input_dim=FEATURE_DIM, d_model=D_MODEL, nhead=N_HEAD, 
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, 
        output_dim=FEATURE_DIM, pred_len=PRED_LEN, dropout=DROPOUT
    ).to(device)
    
    MODEL_PATH = "weather_transformer_best.pth"
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    warmup_epochs = 2
    scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
    
    scaler = torch.amp.GradScaler('cuda')

    with mlflow.start_run():
        logger.info("Starting Transformer training loop...")
        
        mlflow.log_params({
            "model_type": "Transformer",
            "seq_len": SEQ_LEN, "pred_len": PRED_LEN, "batch_size": BATCH_SIZE,
            "d_model": D_MODEL, "n_heads": N_HEAD, "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "initial_lr": LEARNING_RATE, "optimizer": "AdamW", "loss": "SmoothL1Loss"
        })

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(EPOCHS):
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
            val_mae = val_mae_sum / len(val_loader)
            val_rmse = math.sqrt(val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"--- Epoch {epoch+1} Complete | Train Loss: {train_loss:.4f} | Val Loss (SmoothL1): {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} | LR: {current_lr:.6f} ---")
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "learning_rate": current_lr
            }, step=epoch)

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), MODEL_PATH)
                logger.info(f"New best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")

            if epochs_without_improvement >= PATIENCE:
                logger.warning(f"Early stopping triggered after {epoch+1} epochs to prevent overfitting.")
                break

        logger.info("Loading best weights and logging full model artifact to MLflow...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        model.to("cpu")
        sample_input, _ = next(iter(val_loader))
        
        mlflow.pytorch.log_model(
            model, 
            name="weather_transformer_model",
            input_example=sample_input.numpy(),
            serialization_format="pickle"
        )
            
        logger.info("MLflow model logging completed.")

if __name__ == "__main__":
    main()