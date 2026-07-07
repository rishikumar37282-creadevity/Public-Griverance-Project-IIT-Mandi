"""Text -> integer sequences (Layer 0 of the architecture).

Tokenize (lowercased whitespace split), build vocabulary with <PAD>=0 and
<UNK>=1, encode + pad to max_len. Mirrors notebook section 5.
"""
from collections import Counter

import pandas as pd

from .config import Config, URGENCY_LABELS
from .urgency_oracle import urgency_score


def tokenize(s: str, lowercase: bool = True):
    return (s.lower() if lowercase else s).split()


def build_vocab(texts, cfg: Config):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t, cfg.lowercase))
    itos = ["<PAD>", "<UNK>"] + [w for w, c in counter.most_common(cfg.max_vocab) if c >= cfg.min_freq]
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def encode(text: str, stoi: dict, cfg: Config):
    ids = [stoi.get(tok, 1) for tok in tokenize(text, cfg.lowercase)][: cfg.max_len]  # 1 = <UNK>
    return ids + [0] * (cfg.max_len - len(ids))                                       # 0 = <PAD>


def load_dataframe(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)
    tcol = cfg.text_col if cfg.text_col in df.columns else max(
        df.columns, key=lambda c: df[c].astype(str).str.len().mean())
    df = df.rename(columns={tcol: "text"})
    if "category" not in df.columns:
        df["category"] = "Unknown"
    if "urgency" not in df.columns:
        df["urgency"] = df["text"].apply(lambda t: urgency_score(str(t))[1])
    return df.dropna(subset=["text"]).reset_index(drop=True)


def prepare(df: pd.DataFrame, cfg: Config):
    """Attach encoded x / y columns; return (df, itos, stoi, cat_labels)."""
    itos, stoi = build_vocab(df["text"], cfg)
    cat_labels = sorted(df["category"].unique())
    cat2i = {c: i for i, c in enumerate(cat_labels)}
    urg2i = {u: i for i, u in enumerate(URGENCY_LABELS)}
    df = df.copy()
    df["x"] = df["text"].apply(lambda t: encode(t, stoi, cfg))
    df["y_cat"] = df["category"].map(cat2i)
    df["y_urg"] = df["urgency"].map(urg2i)
    return df, itos, stoi, cat_labels
