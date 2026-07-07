"""FastAPI backend serving both platforms:

  /            -> citizen portal (Jan Samadhan)
  /admin       -> Model Observatory (admin dashboard)
  /api/...     -> citizen APIs (classify, track, stats)
  /api/admin/… -> model card, architecture explainer trace, live training
                  studio, experiment history from logs/training_log.csv

Run:  uvicorn app.server:app --reload   (from the project root)
"""
import csv
import json
import random
import re
import string
import sys
import threading
import datetime
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, TUNABLE_HYPERPARAMETERS, URGENCY_LABELS, DATA_DIR, LOGS_DIR  # noqa: E402
from src.inference import get_classifier, reload_classifier  # noqa: E402
from src.escalation import recommend  # noqa: E402
from src.urgency_oracle import urgency_score  # noqa: E402
from src.train import train_run  # noqa: E402

app = FastAPI(title="Public Grievance Urgency Classification System",
              description="AI-assisted triage of citizen grievances: category, department, urgency and escalation.",
              version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
COMPLAINTS_DB = DATA_DIR / "complaints_db.json"
_db_lock = threading.Lock()

# ---------------------------------------------------------------- category tree
with open(DATA_DIR / "category_tree.json", encoding="utf-8") as f:
    CATEGORY_TREE = json.load(f)

_STOPWORDS = {"the", "a", "an", "of", "to", "in", "for", "and", "or", "not", "no", "my", "is",
              "are", "related", "others", "other", "complaint", "complaints", "regarding"}


def suggest_subcategories(text: str, org_code: str | None, k: int = 3):
    """Rank CPGRAMS sub-categories of the predicted organisation by keyword overlap."""
    if not org_code or org_code not in CATEGORY_TREE:
        return []
    words = {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}
    scored = []
    for node in CATEGORY_TREE[org_code]:
        if node["stage"] < 2:
            continue
        desc_words = {w for w in re.findall(r"[a-z]+", node["desc"].lower())
                      if w not in _STOPWORDS and len(w) > 2}
        overlap = words & desc_words
        if overlap:
            # deeper nodes are more specific -> small bonus per stage
            scored.append((len(overlap) + 0.3 * node["stage"], node, sorted(overlap)))
    scored.sort(key=lambda t: -t[0])
    return [{"code": n["code"], "description": n["desc"], "stage": n["stage"], "matched": m}
            for _, n, m in scored[:k]]


# ---------------------------------------------------------------- complaint store
def _load_db():
    if COMPLAINTS_DB.exists():
        with open(COMPLAINTS_DB, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_db(db):
    with open(COMPLAINTS_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=1, ensure_ascii=False)


def _new_reference(db):
    while True:
        ref = "PG-" + datetime.date.today().strftime("%Y") + "-" + \
              "".join(random.choices(string.digits, k=6))
        if ref not in db:
            return ref


# ---------------------------------------------------------------- request models
class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ComplaintRequest(BaseModel):
    text: str = Field(min_length=10, max_length=4000)
    name: str = ""
    location: str = ""
    contact: str = ""


class TrainRequest(BaseModel):
    overrides: dict = {}
    run_name: str = ""
    promote: bool = False


# ---------------------------------------------------------------- citizen APIs
@app.post("/api/analyze")
def analyze_draft(req: AnalyzeRequest):
    """Live triage preview while the citizen is typing (does not file anything)."""
    clf = get_classifier()
    res = clf.classify(req.text)
    return {
        "category": res["category"], "category_confidence": res["category_confidence"],
        "department": res["department"], "urgency": res["urgency"],
        "urgency_probs": res["urgency_probs"], "urgency_score": res["urgency_score"],
        "score_breakdown": res["score_breakdown"], "matched_signals": res["matched_signals"],
        "tokens": res["tokens"], "attention_norm": res["attention_norm"],
    }


@app.post("/api/classify")
def classify_and_file(req: ComplaintRequest):
    """Full classification + escalation plan; files the complaint and returns a reference ID."""
    clf = get_classifier()
    res = clf.classify(req.text)
    plan = recommend(res["urgency"], res["department"], res["org_code"])
    subcats = suggest_subcategories(req.text, res["org_code"])

    with _db_lock:
        db = _load_db()
        ref = _new_reference(db)
        db[ref] = {
            "reference_id": ref,
            "filed_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "text": req.text, "name": req.name, "location": req.location, "contact": req.contact,
            "category": res["category"], "department": res["department"],
            "org_code": res["org_code"], "category_code": res["category_code"],
            "urgency": res["urgency"], "urgency_score": res["urgency_score"],
            "status": "Registered", "first_response_due": plan["first_response"],
            "resolution_target": plan["resolution_target"],
            "subcategories": subcats,
        }
        _save_db(db)

    return {"reference_id": ref, "filed_at": db[ref]["filed_at"],
            "classification": res, "action_plan": plan, "subcategories": subcats}


@app.get("/api/complaints/{reference_id}")
def track_complaint(reference_id: str):
    db = _load_db()
    rec = db.get(reference_id.strip().upper())
    if not rec:
        raise HTTPException(404, "No complaint found for this reference ID.")
    # never echo back personal contact data on the public track endpoint
    return {k: v for k, v in rec.items() if k not in ("contact",)}


@app.get("/api/stats")
def public_stats():
    db = _load_db()
    urg = Counter(r["urgency"] for r in db.values())
    dept = Counter(r["department"] for r in db.values())
    return {"total": len(db),
            "by_urgency": {u: urg.get(u, 0) for u in URGENCY_LABELS},
            "by_department": dict(dept.most_common(10))}


@app.get("/api/departments")
def departments():
    clf = get_classifier()
    return [{"category": c, **clf.departments[c]} for c in clf.cat_labels if c in clf.departments]


# ---------------------------------------------------------------- admin: model card & explainer
@app.get("/api/admin/model")
def model_card():
    return get_classifier().model_card()


@app.post("/api/admin/explain")
def explain(req: AnalyzeRequest):
    return get_classifier().explain_trace(req.text)


@app.get("/api/admin/hyperparameters")
def hyperparameters():
    return TUNABLE_HYPERPARAMETERS


@app.get("/api/admin/dataset")
def dataset_stats():
    import pandas as pd
    df = pd.read_csv(Config().csv_path)
    lengths = df["text"].str.split().str.len()
    hist, edges = [], list(range(0, 26, 2))
    for lo, hi in zip(edges[:-1], edges[1:]):
        hist.append(int(((lengths >= lo) & (lengths < hi)).sum()))
    return {
        "n_rows": int(len(df)),
        "by_urgency": df["urgency"].value_counts().reindex(URGENCY_LABELS).fillna(0).astype(int).to_dict(),
        "by_category": df["category"].value_counts().astype(int).to_dict(),
        "length_hist": {"edges": edges, "counts": hist},
        "length_mean": round(float(lengths.mean()), 1),
        "urgency_score_mean": round(float(df["urgency_score"].mean()), 1) if "urgency_score" in df else None,
    }


# ---------------------------------------------------------------- admin: training studio
class _TrainerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.stop_event = threading.Event()
        self.reset()

    def reset(self):
        self.running = False
        self.run_name = ""
        self.promote = False
        self.overrides = {}
        self.epochs = []
        self.summary = None
        self.error = None
        self.started_at = None


TRAINER = _TrainerState()


def _train_worker(overrides, run_name, promote):
    try:
        summary = train_run(
            overrides,
            epoch_callback=lambda row: TRAINER.epochs.append(row),
            stop_flag=TRAINER.stop_event.is_set,
            save_as_production=promote,
            run_name=run_name,
        )
        TRAINER.summary = summary
        if promote:
            reload_classifier()
    except Exception as e:  # surface the error to the dashboard instead of dying silently
        TRAINER.error = f"{type(e).__name__}: {e}"
    finally:
        TRAINER.running = False


@app.post("/api/admin/train")
def start_training(req: TrainRequest):
    with TRAINER.lock:
        if TRAINER.running:
            raise HTTPException(409, "A training run is already in progress.")
        # validate overrides against the tunable schema
        clean = {}
        for k, v in req.overrides.items():
            spec = TUNABLE_HYPERPARAMETERS.get(k)
            if spec is None:
                continue
            if spec["type"] == "choice":
                if v not in spec["choices"]:
                    raise HTTPException(422, f"{k}: must be one of {spec['choices']}")
            elif spec["type"] == "bool":
                v = bool(v) if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            else:
                try:
                    v = float(v) if spec["type"] == "float" else int(v)
                except (TypeError, ValueError):
                    raise HTTPException(422, f"{k}: invalid number")
                if not (spec["min"] <= v <= spec["max"]):
                    raise HTTPException(422, f"{k}: must be between {spec['min']} and {spec['max']}")
            clean[k] = v
        TRAINER.reset()
        TRAINER.stop_event.clear()
        TRAINER.running = True
        TRAINER.run_name = req.run_name or f"studio-{datetime.datetime.now().strftime('%H%M%S')}"
        TRAINER.promote = req.promote
        TRAINER.overrides = clean
        TRAINER.started_at = datetime.datetime.now().isoformat(timespec="seconds")
        TRAINER.thread = threading.Thread(
            target=_train_worker, args=(clean, TRAINER.run_name, req.promote), daemon=True)
        TRAINER.thread.start()
    return {"started": True, "run_name": TRAINER.run_name, "overrides": clean, "promote": req.promote}


@app.get("/api/admin/train/status")
def training_status(since_epoch: int = 0):
    return {
        "running": TRAINER.running,
        "run_name": TRAINER.run_name,
        "promote": TRAINER.promote,
        "overrides": TRAINER.overrides,
        "started_at": TRAINER.started_at,
        "total_epochs_planned": TRAINER.overrides.get("epochs", Config().epochs),
        "epochs": TRAINER.epochs[since_epoch:],
        "epoch_count": len(TRAINER.epochs),
        "summary": ({k: v for k, v in TRAINER.summary.items() if k != "config"}
                    if TRAINER.summary else None),
        "error": TRAINER.error,
    }


@app.post("/api/admin/train/stop")
def stop_training():
    if not TRAINER.running:
        raise HTTPException(409, "No training run in progress.")
    TRAINER.stop_event.set()
    return {"stopping": True}


# ---------------------------------------------------------------- admin: experiment history
def _read_log_rows():
    log_path = Path(Config().log_csv_path)
    if not log_path.exists():
        return []
    with open(log_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# everything else in the log is numeric — new metric/hyperparameter columns are
# picked up automatically without another schema list to maintain
STRING_COLS = {"run_id", "run_name", "timestamp", "optimizer", "scheduler",
               "use_class_weights", "device", "bidirectional", "lowercase"}


def _typed(row):
    out = {}
    for k, v in row.items():
        if k in STRING_COLS or v in ("", None):
            out[k] = v
        else:
            try:
                out[k] = float(v) if "." in str(v) or "e" in str(v).lower() else int(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


HPARAM_COLS = ["emb_dim", "hidden_dim", "num_layers", "bidirectional", "attn_dim", "dropout",
               "head_hidden", "max_len", "lowercase", "batch_size", "optimizer", "scheduler",
               "weight_decay", "grad_clip", "patience", "critical_boost", "w_category",
               "w_urgency", "use_class_weights", "seed"]


@app.get("/api/admin/runs")
def list_runs():
    rows = [_typed(r) for r in _read_log_rows()]
    runs = {}
    for r in rows:
        runs.setdefault(r["run_id"], []).append(r)
    out = []
    for run_id, epochs in runs.items():
        epochs.sort(key=lambda r: r["epoch"])
        last, best = epochs[-1], max(epochs, key=lambda r: r["val_critical_recall"])
        out.append({
            "run_id": run_id, "run_name": last.get("run_name", run_id),
            "n_epochs": len(epochs),
            "hyperparameters": {k: last.get(k) for k in HPARAM_COLS},
            "final": {k: last.get(k) for k in ("train_loss", "val_loss", "val_acc", "val_macro_f1",
                                               "val_critical_recall", "val_cat_acc")},
            "best_val_critical_recall": best["val_critical_recall"],
            "best_epoch": best["epoch"],
            "total_seconds": round(sum(e.get("epoch_seconds", 0) for e in epochs), 1),
        })
    out.sort(key=lambda r: r["run_id"], reverse=True)
    return out


@app.get("/api/admin/runs/{run_id}")
def run_detail(run_id: str):
    rows = [_typed(r) for r in _read_log_rows() if r["run_id"] == run_id]
    if not rows:
        raise HTTPException(404, "Unknown run_id.")
    rows.sort(key=lambda r: r["epoch"])
    return {"run_id": run_id, "run_name": rows[-1].get("run_name", run_id), "epochs": rows}


@app.get("/api/admin/log/download")
def download_log():
    log_path = Path(Config().log_csv_path)
    if not log_path.exists():
        raise HTTPException(404, "No training log yet.")
    return FileResponse(log_path, media_type="text/csv", filename="training_log.csv")


# ---------------------------------------------------------------- oracle-only endpoint (guardrail/fallback)
@app.post("/api/oracle")
def oracle_only(req: AnalyzeRequest):
    total, label, parts, matches = urgency_score(req.text)
    return {"urgency_score": total, "label": label, "breakdown": parts, "matches": matches}


# ---------------------------------------------------------------- static frontends
@app.get("/admin")
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/explainer")
def explainer_page():
    return FileResponse(STATIC_DIR / "explainer.html")


@app.get("/story")
def story_page():
    return FileResponse(STATIC_DIR / "story.html")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
