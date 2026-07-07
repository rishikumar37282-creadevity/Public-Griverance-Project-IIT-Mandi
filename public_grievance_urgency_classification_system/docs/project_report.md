# Project Report — Public Grievance Urgency Classification System

**Track:** AIML · **Difficulty:** Intermediate · **Duration:** 10 days · **Mode:** Individual/Group

---

## 1. Problem understanding and stakeholders

Public grievance systems (e.g. CPGRAMS) receive lakhs of complaints. Everything lands in one
queue, so triage speed depends on manual reading. The cost of slow triage is asymmetric: a late
reply to a suggestion is harmless; a late reply to *"hospital demanded bribe to admit my
critically ill child"* is not.

**Stakeholders**

| Who | Need | What the system gives them |
|---|---|---|
| Citizens | File once, reach the right desk, know what happens next | Auto-routing, urgency score, action plan, reference ID + tracking |
| Grievance officers | See emergencies first | 4-level priority queue with Critical flagged in 24 h |
| Department admins / ML team | Trust and improve the model | Model Observatory: metrics, explainability, retraining, experiment history |

**Success criteria:** high **Critical-class recall** (missing an emergency is the worst error),
reasonable macro-F1, correct department routing, and outputs a non-technical user can act on.

## 2. Data preparation

- `data/grievances.csv`: 4,840 complaints, 7 columns (`text, category, department,
  category_code, org_code, urgency, urgency_score`). 15 categories map 1:1 to departments and
  CPGRAMS org codes. Urgency distribution: Medium 1,873 · Low 1,673 · High 1,177 · **Critical
  117 (2.4%)**. Mean length 12.7 tokens (max 23) → `max_len=40` truncates nothing.
- `data/CategoryCode_Mapping.xlsx`: official CPGRAMS category tree — 19,853 nodes
  (`Code, Description, OrgCode, Parent, Stage`). We extract the 3,899 nodes of our 15
  organisations into `data/category_tree.json`; the app suggests the deepest matching
  sub-category for each complaint by keyword overlap (e.g. a pension complaint surfaces
  *"Atal Pension Yojana"*).
- **Urgency labels are weak supervision**: a transparent rule oracle
  (`src/urgency_oracle.py`) sums five explainable signals — category severity (0–22),
  severity lexicon hits (0–30), pending-duration mentions (0–20), vulnerable-group mentions
  (0/10), escalation depth (0–15) — into a 0–100 score with thresholds 25/45/70.

## 3. Model architecture (DL Architecture 3 — BiLSTM with attention)

```
input ids (B, 40)
 → Embedding(vocab≈406, emb_dim=128, pad=0)         52k params
 → BiLSTM(hidden=128 ×2 dirs, layers=1)             264k params
 → Additive attention (attn_dim=64) → α weights      16k params
 → context vector (256)
 → Head A: MLP 256→128→128→15  (category)            51k params
 → Head B: MLP 256→128→128→4   (urgency)             50k params
                                        total ≈ 434,963 params
```

Both heads use two hidden layers (128 neurons, BatchNorm + ReLU + dropout). The attention
module returns the per-token α weights used for explainability in both apps.

**Loss** = `w_category·CE(category) + w_urgency·CE(urgency, class_weights)` where urgency class
weights are inverse-frequency with the Critical weight multiplied by `critical_boost=3`.

**Training**: AdamW (lr 1e-3, wd 1e-5), ReduceLROnPlateau on val Critical-recall, gradient
clipping 5.0, batch 64, up to 20 epochs, early stopping patience 10. Checkpoint selection:
best `(val_critical_recall, val_macro_f1)` — the tie-break prevents degenerate
"everything-is-Critical" checkpoints. Split 70/15/15 stratified on urgency.

Every epoch of every run is appended to `logs/training_log.csv` — a complete, showcase-ready
training record with 66 columns per epoch: run id/name/timestamp, cost function (train/val
loss), accuracy and macro-F1 for both splits, Critical-recall, **category-head accuracy and
macro-F1**, **per-class precision/recall/F1 for all four urgency classes**, the learning rate,
the **full hyperparameter snapshot** (architecture, data, optimiser, loss weighting, seed,
device), epoch wall-time, and the flattened 4×4 validation confusion matrix. The logger
auto-migrates older files when the schema grows, so history is never lost. This file drives
the admin Experiments view and can be opened directly in Excel/pandas for the report.

## 4. Results

**Held-out test set (726 complaints):**

| Metric | baseline-v1 (production) | no-critical-boost | small-64-highdrop |
|---|---|---|---|
| Critical recall | **1.000** | 0.941 | 1.000 |
| Urgency macro-F1 | 0.886 | 0.908 | 0.674 |
| Urgency accuracy | 87.3% | ~93% | lower |
| Category accuracy | 100.0% | 100.0% | 100.0% |

Per-class (baseline): the Critical row of the confusion matrix has zero misses; residual errors
are adjacent-class (Low↔Medium, Medium↔High), which is acceptable for queue ordering.

**Interpretation of the ablation:** removing class weights/boost trades ~2 points of macro-F1
for **6% of emergencies missed** — exactly the asymmetric error the design guards against.

## 5. Decision-support output (beyond code execution)

For each complaint the system produces: reference ID · department + CPGRAMS codes + sub-category
suggestions · urgency label with probabilities and 0–100 score breakdown · attention-highlighted
text ("why") · routing level, first-response and resolution SLAs · officer escalation path ·
concrete citizen next steps. This is what makes the output *actionable* rather than a bare label.

## 6. Validation

- Metric-based: classification report, confusion matrices, calibration histogram (admin →
  Evaluation), Critical-recall as the headline.
- Scenario checks: pension stoppage (→ Pension, Critical), hospital bribe (→ Health, Critical),
  post-office suggestion (→ Posts, Low), ATM debit dispute (→ Banking) — all verified in the UI.
- Robustness: unknown words map to `<UNK>` and are listed; empty/short inputs are gated in the
  UI; the API validates hyperparameter ranges before training.

## 7. Both platforms (implementation)

- **Backend** `app/server.py` — FastAPI; citizen endpoints (`/api/analyze`, `/api/classify`,
  `/api/complaints/{id}`, `/api/stats`, `/api/departments`) and admin endpoints (model card,
  forward-pass trace, hyperparameter schema, threaded training with live status polling,
  stop, run history, log download). Complaints persist to a JSON store.
- **Citizen portal** — vanilla JS + a custom SVG chart library (gauge, probability bars);
  live triage preview debounced at ~550 ms; Web Speech API voice input; print-friendly
  acknowledgement.
- **Model Observatory** — six views (Overview, Architecture Explorer, Training Studio,
  Experiments, Evaluation, Dataset). The Architecture Explorer renders a real forward pass:
  token ids, embedding heat cells, per-direction LSTM activation norms, attention bars, the
  context vector, and both heads' softmax outputs, with cross-stage hover highlighting —
  UX inspired by the Transformer Explainer (Polo Club, Georgia Tech).

## 8. Limitations & responsible use

See `docs/limitations_responsible_use.md` (summarised in the README and shown inside both apps).

## 9. Future work

MuRIL/IndicBERT Tier-2 model for multilingual complaints; officer feedback loop; duplicate
detection; authenticated roles; deployment hardening (DB, HTTPS, audit logs).
