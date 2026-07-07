"""Train/val/test split (70/15/15, stratified on urgency) and torch Datasets."""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from .config import Config, URGENCY_LABELS


class GrievanceDataset(Dataset):
    def __init__(self, frame):
        self.x = torch.tensor(np.stack(frame["x"].values), dtype=torch.long)
        self.yc = torch.tensor(frame["y_cat"].values, dtype=torch.long)
        self.yu = torch.tensor(frame["y_urg"].values, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.yc[i], self.yu[i]


def split_frames(df, cfg: Config):
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=cfg.seed, stratify=df["y_urg"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=cfg.seed, stratify=temp_df["y_urg"])
    return train_df, val_df, test_df


def make_loaders(train_df, val_df, test_df, cfg: Config):
    train_dl = DataLoader(GrievanceDataset(train_df), batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(GrievanceDataset(val_df), batch_size=cfg.batch_size)
    test_dl = DataLoader(GrievanceDataset(test_df), batch_size=cfg.batch_size)
    return train_dl, val_dl, test_dl


def urgency_class_weights(train_df, cfg: Config):
    """Inverse-frequency weights with an extra boost on the rare Critical class."""
    n_urg = len(URGENCY_LABELS)
    counts = train_df["y_urg"].value_counts().reindex(range(n_urg)).fillna(1).values.astype(float)
    inv = counts.sum() / (n_urg * counts)
    inv[URGENCY_LABELS.index("Critical")] *= cfg.critical_boost
    return torch.tensor(inv, dtype=torch.float, device=cfg.device)
