"""Training loop with per-epoch CSV experiment logging, early stopping on
Critical-recall, and an optional callback for live monitoring (used by the
admin dashboard to stream epochs to the browser).

Every epoch row records: run metadata, losses, accuracy, macro-F1,
Critical-recall, LR, the active hyperparameters, and the flattened 4x4
validation confusion matrix — exactly the schema of the notebook's
training_log.csv, plus a few extra hyperparameter columns.
"""
import csv
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, recall_score, confusion_matrix,
                             precision_recall_fscore_support)

from .config import Config, URGENCY_LABELS, MODELS_DIR, LOGS_DIR
from .model import BiLSTMAttn, parameter_summary
from .preprocessing import load_dataframe, prepare
from .data_utils import split_frames, make_loaders, urgency_class_weights

N_URG = len(URGENCY_LABELS)
CM_HEADERS = [f"cm_{i}_{j}" for i in range(N_URG) for j in range(N_URG)]
# per-epoch, per-class metrics for the urgency head (Low/Medium/High/Critical)
PER_CLASS_HEADERS = [f"val_{m}_{lbl.lower()}" for lbl in URGENCY_LABELS
                     for m in ("prec", "rec", "f1")]
LOG_HEADERS = (["run_id", "run_name", "timestamp", "epoch",
                # cost function + core performance, both splits
                "train_loss", "val_loss", "train_acc", "val_acc",
                "train_macro_f1", "val_macro_f1", "val_critical_recall",
                # category (department) head performance
                "train_cat_acc", "val_cat_acc", "val_cat_macro_f1"]
               + PER_CLASS_HEADERS +
               ["lr",
                # full hyperparameter snapshot for the run
                "emb_dim", "hidden_dim", "num_layers", "bidirectional", "attn_dim",
                "dropout", "head_hidden",
                "max_len", "max_vocab", "min_freq", "lowercase", "batch_size", "epochs_planned",
                "optimizer", "scheduler", "weight_decay", "grad_clip",
                "patience", "critical_boost", "w_category", "w_urgency",
                "use_class_weights", "seed", "device", "epoch_seconds"]
               + CM_HEADERS)


def _ensure_log_schema(log_path: Path):
    """If an existing log uses an older column set, migrate it in place.

    Old rows keep their values; columns they never had are left empty. This keeps
    the full experiment history readable by the admin dashboard across schema
    upgrades instead of corrupting the CSV with mismatched rows.
    """
    if not log_path.exists() or log_path.stat().st_size == 0:
        return
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == LOG_HEADERS:
            return
        old_rows = list(reader)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_HEADERS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(old_rows)
    print(f"Migrated {log_path.name} to the new logging schema ({len(old_rows)} historic rows kept).")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg: Config):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)


def build_scheduler(opt, cfg: Config):
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    return None


def _epoch_pass(model, loader, cfg, loss_fn, opt=None):
    training = opt is not None
    model.train() if training else model.eval()
    total_loss, ys_u, ps_u, ys_c, ps_c = 0.0, [], [], [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, yc, yu in loader:
            x, yc, yu = x.to(cfg.device), yc.to(cfg.device), yu.to(cfg.device)
            if training:
                opt.zero_grad()
            cat_logits, urg_logits, _ = model(x)
            loss = loss_fn(cat_logits, urg_logits, yc, yu)
            if training:
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            total_loss += loss.item()
            ys_u.extend(yu.cpu().numpy())
            ps_u.extend(urg_logits.argmax(1).cpu().numpy())
            ys_c.extend(yc.cpu().numpy())
            ps_c.extend(cat_logits.argmax(1).cpu().numpy())
    return (total_loss / max(len(loader), 1),
            np.array(ys_u), np.array(ps_u), np.array(ys_c), np.array(ps_c))


def train_run(overrides: dict | None = None,
              epoch_callback=None,
              stop_flag=None,
              save_as_production: bool = True,
              run_name: str = ""):
    """Execute one full training run. Returns a summary dict.

    epoch_callback(epoch_row: dict) is called after every epoch (live monitoring).
    stop_flag: callable returning True to abort gracefully.
    """
    cfg = Config.from_overrides(overrides)
    set_seed(cfg.seed)

    df = load_dataframe(cfg)
    df, itos, stoi, cat_labels = prepare(df, cfg)
    train_df, val_df, test_df = split_frames(df, cfg)
    train_dl, val_dl, test_dl = make_loaders(train_df, val_df, test_df, cfg)

    n_cat = len(cat_labels)
    model = BiLSTMAttn(cfg, len(itos), n_cat, N_URG).to(cfg.device)
    layer_rows, n_params = parameter_summary(model)

    class_w = urgency_class_weights(train_df, cfg)
    ce_cat = nn.CrossEntropyLoss()
    ce_urg = nn.CrossEntropyLoss(weight=class_w if cfg.use_class_weights else None)

    def total_loss(cat_logits, urg_logits, yc, yu):
        return cfg.w_category * ce_cat(cat_logits, yc) + cfg.w_urgency * ce_urg(urg_logits, yu)

    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = run_name or cfg.run_name or f"run-{run_id}"
    log_path = Path(cfg.log_csv_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_log_schema(log_path)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    crit_idx = URGENCY_LABELS.index("Critical")

    history = []
    # Checkpoint selection: Critical-recall first (the metric that matters),
    # macro-F1 as tie-break — otherwise an epoch that predicts everything
    # Critical (recall 1.0, terrible F1) would win.
    best_key, best_state, epochs_no_improve, stopped_early = (-1.0, -1.0), None, 0, False

    for epoch in range(1, cfg.epochs + 1):
        if stop_flag and stop_flag():
            stopped_early = True
            break
        t0 = time.time()
        train_loss, tr_y, tr_p, tr_yc, tr_pc = _epoch_pass(model, train_dl, cfg, total_loss, opt)
        val_loss, va_y, va_p, va_yc, va_pc = _epoch_pass(model, val_dl, cfg, total_loss)

        row = {
            "run_id": run_id, "run_name": run_name,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5),
            "train_acc": round(float(accuracy_score(tr_y, tr_p)), 5),
            "val_acc": round(float(accuracy_score(va_y, va_p)), 5),
            "train_macro_f1": round(float(f1_score(tr_y, tr_p, average="macro", zero_division=0)), 5),
            "val_macro_f1": round(float(f1_score(va_y, va_p, average="macro", zero_division=0)), 5),
            "val_critical_recall": round(float(recall_score(va_y, va_p, labels=[crit_idx], average="macro", zero_division=0)), 5),
            "train_cat_acc": round(float(accuracy_score(tr_yc, tr_pc)), 5),
            "val_cat_acc": round(float(accuracy_score(va_yc, va_pc)), 5),
            "val_cat_macro_f1": round(float(f1_score(va_yc, va_pc, average="macro", zero_division=0)), 5),
            "lr": opt.param_groups[0]["lr"],
            "emb_dim": cfg.emb_dim, "hidden_dim": cfg.hidden_dim, "num_layers": cfg.num_layers,
            "bidirectional": cfg.bidirectional,
            "attn_dim": cfg.attn_dim, "dropout": cfg.dropout, "head_hidden": cfg.head_hidden,
            "max_len": cfg.max_len, "max_vocab": cfg.max_vocab, "min_freq": cfg.min_freq,
            "lowercase": cfg.lowercase,
            "batch_size": cfg.batch_size, "epochs_planned": cfg.epochs,
            "optimizer": cfg.optimizer, "scheduler": cfg.scheduler,
            "weight_decay": cfg.weight_decay, "grad_clip": cfg.grad_clip,
            "patience": cfg.patience, "critical_boost": cfg.critical_boost,
            "w_category": cfg.w_category, "w_urgency": cfg.w_urgency,
            "use_class_weights": cfg.use_class_weights, "seed": cfg.seed, "device": cfg.device,
            "epoch_seconds": round(time.time() - t0, 2),
        }
        # full per-class picture for the urgency head, every epoch
        prec, rec, f1c, _ = precision_recall_fscore_support(
            va_y, va_p, labels=range(N_URG), zero_division=0)
        for i, lbl in enumerate(URGENCY_LABELS):
            key = lbl.lower()
            row[f"val_prec_{key}"] = round(float(prec[i]), 5)
            row[f"val_rec_{key}"] = round(float(rec[i]), 5)
            row[f"val_f1_{key}"] = round(float(f1c[i]), 5)
        cm = confusion_matrix(va_y, va_p, labels=range(N_URG)).flatten()
        row.update({h: int(v) for h, v in zip(CM_HEADERS, cm)})
        history.append(row)

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)

        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={row['val_acc']:.4f} val_macro_f1={row['val_macro_f1']:.4f} "
              f"crit_recall={row['val_critical_recall']:.4f} lr={row['lr']:.6f}")
        if epoch_callback:
            epoch_callback(row)

        # early stopping on the metric that matters
        key = (row["val_critical_recall"], row["val_macro_f1"])
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping after {epoch} epochs (no Critical-recall improvement for {cfg.patience}).")
                stopped_early = True
                break

        if sched:
            sched.step(row["val_critical_recall"]) if cfg.scheduler == "plateau" else sched.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- held-out test evaluation with the best checkpoint ----
    test_metrics = evaluate(model, test_dl, cfg, cat_labels)

    summary = {
        "run_id": run_id, "run_name": run_name, "epochs_ran": len(history),
        "stopped_early": stopped_early, "best_val_critical_recall": best_key[0],
        "n_params": n_params, "config": cfg.to_dict(), "test": test_metrics,
    }

    if save_as_production and best_state is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / "bilstm_attn.pt")
        meta = {
            "itos": itos, "cat_labels": cat_labels, "urg_labels": URGENCY_LABELS,
            "config": cfg.to_dict(), "run_id": run_id, "run_name": run_name,
            "n_params": n_params, "layer_params": layer_rows,
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "test_metrics": test_metrics,
            "history": history,
        }
        with open(MODELS_DIR / "bilstm_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"Saved production model to {MODELS_DIR / 'bilstm_attn.pt'}")

    return summary


def evaluate(model, loader, cfg: Config, cat_labels):
    """Full test-set picture: per-class metrics, confusion matrices, calibration."""
    from sklearn.metrics import precision_recall_fscore_support
    model.eval()
    ys_u, ps_u, probs_u, ys_c, ps_c = [], [], [], [], []
    with torch.no_grad():
        for x, yc, yu in loader:
            x = x.to(cfg.device)
            cat_logits, urg_logits, _ = model(x)
            p = urg_logits.softmax(1).cpu().numpy()
            probs_u.append(p)
            ps_u.extend(p.argmax(1))
            ys_u.extend(yu.numpy())
            ps_c.extend(cat_logits.argmax(1).cpu().numpy())
            ys_c.extend(yc.numpy())
    ys_u, ps_u = np.array(ys_u), np.array(ps_u)
    probs = np.concatenate(probs_u)
    crit_idx = URGENCY_LABELS.index("Critical")

    prec, rec, f1, sup = precision_recall_fscore_support(ys_u, ps_u, labels=range(N_URG), zero_division=0)
    cm = confusion_matrix(ys_u, ps_u, labels=range(N_URG))
    true_prob = probs[np.arange(len(ys_u)), ys_u]
    correct = ps_u == ys_u
    bins = np.linspace(0, 1, 21)

    return {
        "urgency_accuracy": round(float(accuracy_score(ys_u, ps_u)), 4),
        "urgency_macro_f1": round(float(f1_score(ys_u, ps_u, average="macro", zero_division=0)), 4),
        "critical_recall": round(float(recall_score(ys_u, ps_u, labels=[crit_idx], average="macro", zero_division=0)), 4),
        "category_accuracy": round(float(accuracy_score(ys_c, ps_c)), 4),
        "category_macro_f1": round(float(f1_score(ys_c, ps_c, average="macro", zero_division=0)), 4),
        "per_class": [{"label": URGENCY_LABELS[i], "precision": round(float(prec[i]), 4),
                       "recall": round(float(rec[i]), 4), "f1": round(float(f1[i]), 4),
                       "support": int(sup[i])} for i in range(N_URG)],
        "confusion_matrix": cm.tolist(),
        "calibration": {
            "bins": bins[:-1].round(2).tolist(),
            "correct": np.histogram(true_prob[correct], bins=bins)[0].tolist(),
            "wrong": np.histogram(true_prob[~correct], bins=bins)[0].tolist(),
        },
        "n_test": int(len(ys_u)),
    }
