import os
import torch
import logging
import random
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pyiceberg.catalog import load_catalog

logger = logging.getLogger("ML_Training")

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
        
        table = catalog.load_table(tuple(table_name.split(".")))
        arrow_table = table.scan(selected_fields=("features",)).to_arrow()
        
        col = arrow_table.column("features")
        flattened_data = []
        for chunk in col.chunks:
            flattened_data.extend(chunk.to_pylist())
            
        self.data = torch.tensor(flattened_data, dtype=torch.float32)
        self.total_windows = len(self.data) - self.seq_len - self.pred_len + 1
        logger.info(f"Dataset ready. Generated {self.total_windows} total chronological windows.")

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_len
        target_end = end_idx + self.pred_len
        
        x = self.data[start_idx : end_idx]
        y = self.data[end_idx : target_end]
        return x, y


def get_dataloaders(table_name, seq_len, pred_len, batch_size, is_incremental):
    full_dataset = IcebergTimeSeriesDataset(table_name, seq_len=seq_len, pred_len=pred_len)
    total_windows = len(full_dataset)

    if is_incremental:
        recent_windows = 24 * 14
        recent_indices = list(range(max(0, total_windows - recent_windows), total_windows))
        
        historical_pool = list(range(0, max(0, total_windows - recent_windows)))
        sample_size = int(len(historical_pool) * 0.05)
        replay_indices = random.sample(historical_pool, sample_size) if historical_pool else []
        
        selected_indices = recent_indices + replay_indices
        training_dataset = Subset(full_dataset, selected_indices)
        logger.info(f"Incremental mode: Training on {len(selected_indices)} windows ({recent_windows} recent + {sample_size} historical replay).")
    else:
        training_dataset = full_dataset
        logger.info(f"Scratch mode: Training on ALL {len(full_dataset)} historical windows.")

    train_size = int(0.8 * len(training_dataset))
    val_size = len(training_dataset) - train_size
    train_dataset, val_dataset = random_split(training_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    return train_loader, val_loader