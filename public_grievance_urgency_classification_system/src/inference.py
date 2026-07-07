"""Production inference: load saved artifacts and classify complaints.

Returns everything the citizen portal and admin explainer need: predicted
category + department, urgency label with full probability distribution,
per-token attention weights, the oracle score breakdown, and a stage-by-stage
trace of the network for visualization.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config, MODELS_DIR, DATA_DIR, URGENCY_LABELS
from .model import BiLSTMAttn
from .preprocessing import tokenize, encode
from .urgency_oracle import urgency_score


def _department_lookup():
    """category -> (department, category_code, org_code), from the dataset."""
    df = pd.read_csv(DATA_DIR / "grievances.csv")
    out = {}
    for cat, grp in df.groupby("category"):
        r = grp.iloc[0]
        out[cat] = {"department": r["department"], "category_code": int(r["category_code"]),
                    "org_code": r["org_code"]}
    return out


class GrievanceClassifier:
    def __init__(self, models_dir: Path = MODELS_DIR):
        meta_path = Path(models_dir) / "bilstm_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No trained model at {meta_path}. Run `python -m src.main train` first.")
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.cfg = Config.from_overrides(self.meta["config"])
        self.cfg.device = "cpu" if not torch.cuda.is_available() else self.cfg.device
        self.itos = self.meta["itos"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.cat_labels = self.meta["cat_labels"]
        self.urg_labels = self.meta.get("urg_labels", URGENCY_LABELS)
        self.model = BiLSTMAttn(self.cfg, len(self.itos), len(self.cat_labels), len(self.urg_labels))
        self.model.load_state_dict(torch.load(Path(models_dir) / "bilstm_attn.pt",
                                              map_location=self.cfg.device))
        self.model.to(self.cfg.device).eval()
        self.departments = _department_lookup()

    def model_card(self):
        return {
            "run_id": self.meta.get("run_id"), "run_name": self.meta.get("run_name"),
            "trained_at": self.meta.get("trained_at"), "n_params": self.meta.get("n_params"),
            "layer_params": self.meta.get("layer_params", []),
            "vocab_size": len(self.itos), "categories": self.cat_labels,
            "urgency_labels": self.urg_labels, "config": self.meta.get("config", {}),
            "test_metrics": self.meta.get("test_metrics", {}),
            "history": self.meta.get("history", []),
        }

    def classify(self, text: str):
        """Full classification payload for a single complaint text."""
        toks = tokenize(text, self.cfg.lowercase)[: self.cfg.max_len]
        ids = torch.tensor([encode(text, self.stoi, self.cfg)], device=self.cfg.device)
        with torch.no_grad():
            cat_logits, urg_logits, alpha = self.model(ids)
            cat_probs = cat_logits.softmax(1)[0].cpu().numpy()
            urg_probs = urg_logits.softmax(1)[0].cpu().numpy()
        a = alpha[0, : len(toks)].cpu().numpy()
        a_norm = (a / a.max()) if len(a) and a.max() > 0 else a

        cat_i = int(cat_probs.argmax())
        urg_i = int(urg_probs.argmax())
        category = self.cat_labels[cat_i]
        dept = self.departments.get(category, {"department": category, "category_code": None, "org_code": None})

        score, oracle_label, parts, matches = urgency_score(text, category)

        top_cats = sorted(
            ({"category": self.cat_labels[i], "prob": round(float(p), 4)} for i, p in enumerate(cat_probs)),
            key=lambda d: -d["prob"])[:3]

        return {
            "category": category,
            "category_confidence": round(float(cat_probs[cat_i]), 4),
            "top_categories": top_cats,
            "department": dept["department"],
            "category_code": dept["category_code"],
            "org_code": dept["org_code"],
            "urgency": self.urg_labels[urg_i],
            "urgency_confidence": round(float(urg_probs[urg_i]), 4),
            "urgency_probs": {self.urg_labels[i]: round(float(p), 4) for i, p in enumerate(urg_probs)},
            "urgency_score": score,
            "oracle_label": oracle_label,
            "score_breakdown": parts,
            "matched_signals": matches,
            "tokens": toks,
            "attention": [round(float(x), 4) for x in a],
            "attention_norm": [round(float(x), 4) for x in a_norm],
            "unknown_tokens": [t for t in toks if t not in self.stoi],
        }

    def explain_trace(self, text: str, max_tokens: int = 24):
        """Stage-by-stage numeric trace for the admin architecture explainer."""
        toks = tokenize(text, self.cfg.lowercase)[: min(self.cfg.max_len, max_tokens)]
        if not toks:
            toks = ["<empty>"]
        ids_list = [self.stoi.get(t, 1) for t in toks]
        ids_pad = ids_list + [0] * (self.cfg.max_len - len(ids_list))
        x = torch.tensor([ids_pad], device=self.cfg.device)
        with torch.no_grad():
            tr = self.model.forward_trace(x)
        n = len(toks)
        emb = tr["embeddings"][0, :n].cpu().numpy()
        lstm = tr["lstm_out"][0, :n].cpu().numpy()
        h = lstm.shape[1] // 2
        alpha = tr["alpha"][0, :n].cpu().numpy()
        # re-normalise attention over the visible (unpadded) tokens
        if alpha.sum() > 0:
            alpha = alpha / alpha.sum()
        ctx = tr["context"][0].cpu().numpy()
        cat_probs = tr["cat_logits"].softmax(1)[0].cpu().numpy()
        urg_probs = tr["urg_logits"].softmax(1)[0].cpu().numpy()

        def preview(vec, k=8):
            return [round(float(v), 3) for v in vec[:k]]

        return {
            "tokens": toks,
            "token_ids": ids_list,
            "in_vocab": [t in self.stoi for t in toks],
            "embedding_dim": int(emb.shape[1]),
            "embedding_norms": [round(float(np.linalg.norm(e)), 3) for e in emb],
            "embedding_preview": [preview(e) for e in emb],
            "lstm_dim": int(lstm.shape[1]),
            "lstm_fwd_norms": [round(float(np.linalg.norm(v[:h])), 3) for v in lstm],
            "lstm_bwd_norms": [round(float(np.linalg.norm(v[h:])), 3) for v in lstm],
            "attention": [round(float(a), 4) for a in alpha],
            "context_dim": int(ctx.shape[0]),
            "context_preview": preview(ctx, 12),
            "context_norm": round(float(np.linalg.norm(ctx)), 3),
            "category_probs": {self.cat_labels[i]: round(float(p), 4) for i, p in enumerate(cat_probs)},
            "urgency_probs": {self.urg_labels[i]: round(float(p), 4) for i, p in enumerate(urg_probs)},
            "category_logits": [round(float(v), 3) for v in tr["cat_logits"][0].cpu().numpy()],
            "urgency_logits": [round(float(v), 3) for v in tr["urg_logits"][0].cpu().numpy()],
        }


@lru_cache(maxsize=1)
def get_classifier() -> GrievanceClassifier:
    return GrievanceClassifier()


def reload_classifier():
    get_classifier.cache_clear()
    return get_classifier()
