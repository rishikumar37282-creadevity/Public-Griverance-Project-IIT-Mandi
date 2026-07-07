"""CONFIG — every hyperparameter in one place (mirrors the notebook CONFIG cell).

The admin dashboard edits these values per training run; the dataclass makes
overrides explicit and serialisable.
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

URGENCY_LABELS = ["Low", "Medium", "High", "Critical"]


@dataclass
class Config:
    # ---- reproducibility & device ----
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data / preprocessing ----
    max_vocab: int = 20000
    min_freq: int = 2
    max_len: int = 40
    lowercase: bool = True
    csv_path: str = str(DATA_DIR / "grievances.csv")
    text_col: str = "text"

    # ---- model architecture ----
    emb_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    attn_dim: int = 64
    dropout: float = 0.3
    pad_idx: int = 0
    head_hidden: int = 128          # neurons in each of the 2 hidden layers of the MLP heads

    # ---- training ----
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"        # adamw | adam | sgd
    scheduler: str = "plateau"      # plateau | cosine | none
    grad_clip: float = 5.0

    # ---- early stopping ----
    patience: int = 10

    # ---- loss weighting ----
    w_category: float = 1.0
    w_urgency: float = 1.0
    use_class_weights: bool = True
    critical_boost: float = 3.0

    # ---- experiment logging ----
    log_csv_path: str = str(LOGS_DIR / "training_log.csv")
    run_name: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_overrides(cls, overrides: dict | None = None) -> "Config":
        cfg = cls()
        for k, v in (overrides or {}).items():
            if hasattr(cfg, k) and v is not None:
                cur = getattr(cfg, k)
                if isinstance(cur, bool):
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
                elif isinstance(cur, int):
                    v = int(v)
                elif isinstance(cur, float):
                    v = float(v)
                setattr(cfg, k, v)
        return cfg


# Bounds used by the admin UI and API validation so a bad knob cannot crash training.
TUNABLE_HYPERPARAMETERS = {
    "emb_dim":        {"type": "int",    "min": 16,   "max": 512,  "step": 16,   "default": 128,  "group": "Architecture", "help": "Embedding vector size per token. Bigger = richer word features, slower, more prone to overfit."},
    "hidden_dim":     {"type": "int",    "min": 16,   "max": 512,  "step": 16,   "default": 128,  "group": "Architecture", "help": "LSTM hidden units per direction. Raise if underfitting (128→256)."},
    "num_layers":     {"type": "int",    "min": 1,    "max": 4,    "step": 1,    "default": 1,    "group": "Architecture", "help": "Stacked BiLSTM layers. 2 adds depth; dropout applies between layers."},
    "bidirectional":  {"type": "bool",   "default": True,  "group": "Architecture", "help": "Read the text both directions. Off = plain LSTM, halves the context size (H = hidden_dim instead of 2×)."},
    "attn_dim":       {"type": "int",    "min": 8,    "max": 256,  "step": 8,    "default": 64,   "group": "Architecture", "help": "Size of the additive-attention projection."},
    "dropout":        {"type": "float",  "min": 0.0,  "max": 0.8,  "step": 0.05, "default": 0.3,  "group": "Architecture", "help": "Regularisation. Raise (0.3→0.5) if train F1 >> val F1."},
    "head_hidden":    {"type": "int",    "min": 32,   "max": 512,  "step": 32,   "default": 128,  "group": "Architecture", "help": "Neurons in each of the two hidden layers of both MLP heads."},
    "max_len":        {"type": "int",    "min": 10,   "max": 120,  "step": 5,    "default": 40,   "group": "Data",         "help": "Sequence length; complaints are truncated/padded to this many tokens."},
    "max_vocab":      {"type": "int",    "min": 1000, "max": 50000,"step": 1000, "default": 20000,"group": "Data",         "help": "Keep this many most-frequent tokens."},
    "min_freq":       {"type": "int",    "min": 1,    "max": 10,   "step": 1,    "default": 2,    "group": "Data",         "help": "Discard tokens rarer than this."},
    "lowercase":      {"type": "bool",   "default": True, "group": "Data",          "help": "Lower-case text before tokenising. Off keeps case-sensitive tokens (bigger vocab)."},
    "batch_size":     {"type": "int",    "min": 8,    "max": 256,  "step": 8,    "default": 64,   "group": "Training",     "help": "Samples per gradient step. Smaller = noisier but sometimes generalises better."},
    "epochs":         {"type": "int",    "min": 1,    "max": 100,  "step": 1,    "default": 20,   "group": "Training",     "help": "Maximum passes over the training set (early stopping may end sooner)."},
    "lr":             {"type": "float",  "min": 1e-5, "max": 1e-1, "step": 1e-4, "default": 1e-3, "group": "Training",     "help": "Learning rate. Lower (1e-3→3e-4) if loss spikes/NaNs."},
    "weight_decay":   {"type": "float",  "min": 0.0,  "max": 1e-2, "step": 1e-5, "default": 1e-5, "group": "Training",     "help": "L2 regularisation strength."},
    "optimizer":      {"type": "choice", "choices": ["adamw", "adam", "sgd"], "default": "adamw", "group": "Training",     "help": "AdamW is the stable default; SGD+momentum for comparison."},
    "scheduler":      {"type": "choice", "choices": ["plateau", "cosine", "none"], "default": "plateau", "group": "Training", "help": "Plateau halves LR when Critical-recall stalls; cosine decays smoothly."},
    "grad_clip":      {"type": "float",  "min": 0.0,  "max": 10.0, "step": 0.5,  "default": 5.0,  "group": "Training",     "help": "Clip gradient norm; keeps LSTM training stable."},
    "patience":       {"type": "int",    "min": 1,    "max": 30,   "step": 1,    "default": 10,   "group": "Training",     "help": "Early-stopping patience on val Critical-recall."},
    "w_category":     {"type": "float",  "min": 0.0,  "max": 5.0,  "step": 0.1,  "default": 1.0,  "group": "Loss",         "help": "Weight of the category head in the multi-task loss."},
    "w_urgency":      {"type": "float",  "min": 0.0,  "max": 5.0,  "step": 0.1,  "default": 1.0,  "group": "Loss",         "help": "Weight of the urgency head in the multi-task loss."},
    "use_class_weights": {"type": "bool", "default": True, "group": "Loss",       "help": "Inverse-frequency weights protect the rare Critical class."},
    "critical_boost": {"type": "float",  "min": 1.0,  "max": 10.0, "step": 0.5,  "default": 3.0,  "group": "Loss",         "help": "Extra multiplier on the Critical class weight. Raise if Critical-recall is low."},
    "seed":           {"type": "int",    "min": 0,    "max": 9999, "step": 1,    "default": 42,   "group": "Training",     "help": "Random seed for reproducible runs."},
}
