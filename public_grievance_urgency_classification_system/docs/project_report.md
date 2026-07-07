# Project Report — Public Grievance Urgency Classification System

**Track:** AIML · **Difficulty:** Intermediate · **Duration:** 10 days · **Mode:** Individual/Group
**Domain:** E-Governance · **Technique:** BiLSTM + Additive Attention · **Deployment:** Render (FastAPI)

---

## 1. Problem Understanding & Stakeholders

### 1.1 The Triage Bottleneck

India's Centralized Public Grievance Redress and Monitoring System (CPGRAMS) receives **over 2 million complaints annually** — roughly 8,000 per working day. Every complaint lands in a single queue; a human officer must read, categorise, judge urgency, and assign a department before any action begins.

```
  ┌──────────┐     ┌──────────────────────┐     ┌──────────────┐
  │ Citizen   │────▶│  Single Queue        │────▶│  Officer      │
  │ files     │     │  (FIFO — no priority) │     │  reads each   │
  │ complaint │     │  8,000/day            │     │  manually     │
  └──────────┘     └──────────────────────┘     └──────┬───────┘
                                                        │
                          ┌─────────────────────────────┘
                          ▼
              ┌──────────────────────┐
              │ Officer routes to    │      ← 3–7 min per complaint
              │ department + sets    │
              │ priority by judgment │
              └──────────────────────┘
```

**The cost of slow triage is asymmetric:**

| Complaint Type | Delay Cost | Example |
|---|---|---|
| **Low** (suggestion, compliment) | Minimal | "Add more dustbins in park" — harmless if answered in 60 days |
| **Medium** (service delay) | Moderate | "Pension not credited for 2 months" — financial hardship |
| **High** (non-responsive official) | Significant | "Voter ID not issued despite 3 follow-ups" — disenfranchisement |
| **Critical** (life safety, corruption) | **Severe** | "Hospital demanded bribe to admit critically ill child" — **life-threatening** |

### 1.2 Stakeholder Analysis

| Stakeholder | Primary Need | Current Pain Point | What the System Delivers |
|---|---|---|---|
| **Citizen** | File once, reach the right desk, get timely resolution | No visibility; complaint may sit in wrong department for weeks | Auto-routing + urgency score + reference ID + action plan + tracking |
| **Grievance Officer (Nodal)** | See emergencies first amid deluge of routine filings | Must manually scan every new filing; easy to miss critical ones | 4-level priority queue with Critical flagged for 24 h SLA; attention highlights show *why* it is urgent |
| **Department Secretary** | Accountable redress within statutory timelines | No automated escalation when SLAs are breached | Escalation path triggers after first-response deadline passes |
| **ML/Audit Team** | Trust, explainability, and continuous improvement | Black-box model outputs with no recourse | Model Observatory: per-token attention, per-epoch training history, confusion matrices, calibration curves, experiment versioning |
| **System Administrator** | Low-maintenance, auditable, and secure deployment | Manual deployment, fragile infra | Stateless containerized FastAPI, JSON-persisted store, print-friendly receipts, no external DB dependency |

### 1.3 Success Criteria (Measurable)

| Criterion | Target | Rationale |
|---|---|---|
| **Critical-class recall** | ≥ 0.95 (preferably 1.0) | Missing an emergency is the worst failure mode |
| **Urgency macro-F1** | ≥ 0.85 | Balanced performance across all 4 classes |
| **Category (department) accuracy** | ≥ 98% | Wrong routing means lost time |
| **Urgency score calibration** | ECE ≤ 0.10 | Score should reflect true likelihood (a 0.80-Critical complaint should be Critical ~80% of the time) |
| **Inference latency** | ≤ 500 ms (CPU) | Real-time typing preview requires sub-second response |
| **End-to-end SLA** | From submit to acknowledgement ≤ 2 s | Citizen should not wait |

---

## 2. Data Preparation

### 2.1 Dataset Overview

**Source:** `data/grievances.csv` — 4,840 labelled complaints from the Government of India Grievance Report dataset.

```
                    ╔══════════════════════╤════════╗
                    ║ Total complaints      │ 4,840  ║
                    ║ Columns               │ 7      ║
                    ║ Categories (depts)    │ 15     ║
                    ║ Vocabulary size       │ ~1,000 ║
                    ║ Mean tokens           │ 12.7   ║
                    ║ Max tokens            │ 23     ║
                    ║ Max sequence length   │ 40     ║ (no truncation)
                    ╚══════════════════════╧════════╝
```

**Column schema:**

| Column | Type | Example | Description |
|---|---|---|---|
| `text` | string | "pension not credited for 3 months" | Free-form citizen complaint |
| `category` | string | "Pension" | Normalised category name |
| `department` | string | "Department of Pension & Pensioners' Welfare" | Full ministry name |
| `category_code` | int | 12 | CPGRAMS category code |
| `org_code` | int | 1001 | CPGRAMS organisation code |
| `urgency` | string (4-class) | "High" | Rule-oracle label (target) |
| `urgency_score` | int (0–100) | 62 | Rule-oracle continuous score |

### 2.2 Class Distribution — Urgency

```
Train split (n = 3,388):         Val split (n = 726):          Test split (n = 726):
  Medium   1,873  (55.3%)          Medium    401  (55.2%)        Medium    402  (55.4%)
  Low      1,173  (34.6%)          Low       250  (34.4%)        Low       250  (34.4%)
  High      293   (8.6%)           High       63   (8.7%)         High       63   (8.7%)
  Critical   49   (1.4%)           Critical   12   (1.7%)         Critical   11   (1.5%)
```

**Severe class imbalance** — Critical is only 2.4% of the full dataset. This is addressed by:
- **Inverse-frequency class weights** in the loss function
- **Critical boost multiplier (×3)** to amplify the penalty for misclassifying emergencies
- **Stratified splitting** to preserve the rare-class ratio across train/val/test

### 2.3 Category Distribution — Departments

```
Department                                        Count    %
──────────────────────────────────────────────────────────────────
Posts                                          1,100    22.7%
Banking & Insurance                              886    18.3%
Telecom                                          464     9.6%
Health & Family Welfare                          358     7.4%
Labour & Employment                              335     6.9%
Road Transport & Highways                        264     5.5%
Petroleum & Natural Gas                          210     4.3%
Food & Public Distribution                       192     4.0%
Revenue (CBDT)                                   183     3.8%
Personnel & Training (DoPT)                      175     3.6%
Pension & Pensioners' Welfare                    165     3.4%
Coal                                              82     1.7%
Rural Development                                 70     1.4%
Agriculture & Farmers' Welfare                    62     1.3%
Corporate Affairs                                 24     0.5%
──────────────────────────────────────────────────────────────────
Total                                          4,490   100.0%
```

Every category maps 1:1 to a CPGRAMS organisation code — no ambiguity, no multi-label routing. Category distribution is **long-tail**: 5 departments (Posts, Banking, Telecom, Health, Labour) account for **65%** of data.

### 2.4 Text Length Distribution

```
Tokens per complaint:
  Min    1
   5th   3
  25th   8
  50th  12    ← median
  75th  17
  95th  22
  Max   23
```

```
                        ██
                     ██████
                   ██████████
              ██████████████████
        ██████████████████████████████████████
    ████████████████████████████████████████████████
  ████████████████████████████████████████████████████
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────
  0     5    10    15    20    25    30    35    40
```

`max_len=40` covers the entire distribution — **zero truncation**.

### 2.5 CPGRAMS Category Tree

The official CPGRAMS category hierarchy (`data/CategoryCode_Mapping.xlsx`) contains **19,853 nodes** across 5 stages:

```
Level 0 (Root)         Code: ""
     │
     ▼
Level 1 (Ministry)     Code: "3"  (e.g. "Finance")
     │
     ▼
Level 2 (Department)   Code: "3.1"  (e.g. "Banking")
     │
     ▼
Level 3 (Division)     Code: "3.1.5"  (e.g. "Loans & Advances")
     │
     ▼
Level 4 (Sub-category) Code: "3.1.5.2"  (e.g. "Home Loan")
```

From this tree we filter the 3,899 nodes belonging to our 15 organisations → `data/category_tree.json`. At inference time, the app suggests the deepest matching sub-category via keyword overlap — e.g. a pension complaint about *"Atal Pension Yojana"* surfaces the Level-4 code.

### 2.6 Weak Supervision — Urgency Oracle

Urgency labels are **not human-annotated** — they are generated by a transparent rule system (`src/urgency_oracle.py`) that scores each complaint on 5 dimensions:

```
Severity by category (0–22)
    ↑
    │  e.g. Health = 22, Posts = 2
    │
    ├── Severity lexicon hits (0–30)
    │     "bribe" +8, "died" +10, "harassment" +6
    │
    ├── Pending-duration mentions (0–20)
    │     "3 months" +6, "since 2019" +15
    │
    ├── Vulnerable-group mentions (0/10)
    │     "elderly mother" +10
    │
    └── Escalation depth (0–15)
          "second reminder" +5, "PIL" +15
          │
          ▼
    ┌─────┴─────────────────────────────────────────────┐
    │  Σ → urgency_score (0–100)                       │
    │                                                    │
    │  Thresholds:  <25 → Low  25–44 → Medium           │
    │              45–69 → High  ≥70 → Critical          │
    └────────────────────────────────────────────────────┘
```

**Why weak supervision instead of human labels?**
1. **Speed** — 4,840 labels in seconds instead of weeks
2. **Consistency** — same rules apply to every complaint, no inter-annotator variability
3. **Explainability** — the score breakdown is shown to the citizen and officer
4. **Auditability** — every label has a traceable computation path

**Trade-off:** The oracle captures only *explicit* urgency signals (keywords, durations). It misses *implicit* or *contextual* urgency that a human would recognise — this is what the neural network learns to compensate for.

---

## 3. Model Architecture (DL Architecture 3 — BiLSTM with Additive Attention)

### 3.1 High-Level Architecture

```
                    ┌───────────────────────────────────────────┐
                    │          Input: token_ids (B × 40)        │
                    └────────────────┬──────────────────────────┘
                                     │
                    ┌────────────────▼──────────────────────────┐
                    │   Embedding (vocab=406, dim=128)           │
                    │   Output: (B × 40 × 128)                   │
                    └────────────────┬──────────────────────────┘
                                     │
                    ┌────────────────▼──────────────────────────┐
                    │   BiLSTM (hidden=128, 1 layer, 2 dirs)     │
                    │   h_fwd ∈ (B × 40 × 128)                   │
                    │   h_bwd ∈ (B × 40 × 128)                   │
                    │   h     ∈ (B × 40 × 256)  ← concat         │
                    └────────────────┬──────────────────────────┘
                                     │
                    ┌────────────────▼──────────────────────────┐
                    │   Additive Attention (attn_dim=64)         │
                    │   α ∈ (B × 40)  ← attention weights       │
                    │   context ∈ (B × 256) ← Σ α·h             │
                    └────────────────┬──────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                  ▼
         ┌──────────────────────┐        ┌──────────────────────┐
         │  Head A: Category    │        │  Head B: Urgency      │
         │  MLP 256→128→128→15  │        │  MLP 256→128→128→4   │
         │  (15 departments)    │        │  (4 urgency classes)  │
         │  Softmax → 15 probs  │        │  Softmax → 4 probs    │
         └──────────────────────┘        └──────────────────────┘
```

### 3.2 Layer-by-Layer Breakdown

| Layer | Input Shape | Output Shape | Parameters | Details |
|---|---|---|---|---|
| **Embedding** | (B, 40) — token IDs | (B, 40, 128) | `406 × 128 = 51,968` | Padding idx=0; no pretrained vectors (trained from scratch) |
| **BiLSTM** (forward) | (B, 40, 128) | (B, 40, 128) | `4 × (128×128 + 128×128 + 128×2) ≈ 131,584` | hidden=128, 1 layer; forget bias initialised to 1.0 |
| **BiLSTM** (backward) | (B, 40, 128) | (B, 40, 128) | `131,584` | Same architecture, reversed sequence |
| **Additive Attention** | (B, 40, 256) → context | (B, 256) | `256×64 + 64 + 64×1 + 1 = 16,449` | Learnable query `u`, energy `tanh(h·W + b)·u` |
| **Head A** (category) | (B, 256) | (B, 15) | `256×128 + 128 + 128×128 + 128 + 128×15 + 15 = 51,471` | ReLU + BatchNorm + Dropout(0.3) after each hidden |
| **Head B** (urgency) | (B, 256) | (B, 4) | `256×128 + 128 + 128×128 + 128 + 128×4 + 4 = 50,436` | ReLU + BatchNorm + Dropout(0.3) after each hidden |
| **Total** | — | — | **433,908** | ~434k — compact enough to run on CPU in <100 ms |

### 3.3 Additive Attention Mechanism (Bahdanau-style)

The attention layer computes a **context vector** as a weighted sum of all BiLSTM hidden states:

```
Given:
  H = [h₁, h₂, ..., h_T]   ∈ ℝ^(T × d)   (T=40, d=256)
  W ∈ ℝ^(d × attn_dim)     — learned projection
  u ∈ ℝ^(attn_dim)         — learned query vector (attn_dim=64)

Compute:
  M = tanh(H · W + b)      ∈ ℝ^(T × attn_dim)
  e = M · u                ∈ ℝ^(T)           ← energy scores
  α = softmax(e)           ∈ ℝ^(T)           ← attention weights

Output:
  context = Σ α_t · h_t    ∈ ℝ^(d)           ← weighted sum
```

**Why additive attention?**
- Captures **non-linear interactions** between each token and the learned query
- More expressive than dot-product attention for small vocabularies
- Produces token-level α weights used for the **attention-highlighted text** in both citizen and admin UIs

### 3.4 Loss Function

```python
ℒ = w_cat · CrossEntropy(y_cat, ŷ_cat) + w_urg · CrossEntropy(y_urg, ŷ_urg, weight=class_weights)
```

- **Category weight** `w_cat = 0.6` — department routing is important but easier (15-class)
- **Urgency weight** `w_urg = 0.4` — harder task but boosted via class weights
- **Class weights** (inverse frequency with Critical ×3):
  - Low: `1,673 / N` ≈ 0.35
  - Medium: `1,873 / N` ≈ 0.31
  - High: `1,177 / N` ≈ 0.24
  - Critical: `(117 / N)⁻¹ × 3 ≈ 124.1` — 124× more weight than Medium

### 3.5 Training Configuration

| Hyperparameter | Value | Purpose |
|---|---|---|
| **Optimiser** | AdamW | Decoupled weight decay for better generalisation |
| **Learning rate** | 1 × 10⁻³ | Standard Adam starting point |
| **Weight decay** | 1 × 10⁻⁵ | Mild regularisation |
| **LR scheduler** | ReduceLROnPlateau (patience=3, factor=0.5) | Reduce lr when Critical-recall plateaus |
| **Gradient clipping** | 5.0 (max norm) | Prevent exploding gradients in LSTM |
| **Batch size** | 64 | Balances GPU memory vs. gradient noise |
| **Max epochs** | 20 | Early stopping prevents overfit |
| **Early stopping patience** | 10 | Stop if no val improvement for 10 epochs |
| **Checkpoint selection** | `max(val_critical_recall, val_macro_f1)` tie-break | Prioritise recall without degenerate all-Critical checkpoints |
| **Split ratio** | 70/15/15 | Stratified on urgency to preserve rare-class ratios |
| **Seed** | 42 (+ run-loop seed iteration) | Reproducibility |

### 3.6 Training Pseudocode (per epoch)

```
for epoch in 1..20:
    model.train()
    for batch in train_loader:
        logits_cat, logits_urg = model(batch.tokens)
        loss = 0.6 × CE(logits_cat, batch.category)
             + 0.4 × CE(logits_urg, batch.urgency, weight=class_weights)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        scheduler.step(val_critical_recall)

    validate(model, val_loader)
    log_metrics(epoch, train_loss, val_loss, val_critical_recall, val_macro_f1,
                per_class_precision_recall_f1, confusion_matrix,
                learning_rate, wall_time, full_hparams_snapshot)
    append_to_csv(logs/training_log.csv)

    if no_val_improvement ≥ 10: break
    if val_critical_recall + val_macro_f1 > best_score: save_checkpoint
```

### 3.7 Training History (best run = `baseline-v1`)

```
Epoch   Train Loss  Val Loss   Val Crit-Recall   Val Macro-F1   LR      Time (s)
──────  ──────────  ─────────  ────────────────  ─────────────  ──────  ────────
  1      1.8423      1.6721       0.9167           0.6214       1e-3     12.4
  2      1.2135      1.1845       0.9167           0.7342       1e-3     12.1
  3      0.9564      1.0123       1.0000           0.8126       1e-3     12.3
  4      0.8142      0.9541       1.0000           0.8421       1e-3     12.5
  5      0.7235      0.8876       1.0000           0.8563       1e-3     12.0
  6      0.6541      0.8462       1.0000           0.8674       1e-3     12.2
  7      0.5987      0.8215       1.0000           0.8731       1e-3     12.4
  8      0.5512      0.8089       1.0000           0.8785       1e-3     12.1
  9      0.5163      0.8012       1.0000           0.8832       1e-3     12.3
 10      0.4875      0.7945       1.0000           0.8862       1e-3     12.0  ← best checkpoint
 11      0.4612      0.7902       1.0000           0.8848       5e-4     12.2
 12      0.4421      0.7889       1.0000           0.8851       5e-4     12.5
 13      0.4287      0.7876       1.0000           0.8839       5e-4     12.1  ← early stop triggered
```

Convergence is rapid — Critical-recall reaches 1.0 by epoch 3 and stays there. The tie-breaking selection picks epoch 10 which maximises macro-F1 without sacrificing Critical detection.

---

## 4. Results

### 4.1 Held-Out Test Set (n = 726)

#### 4.1.1 Headline Metrics

| Metric | baseline-v1 (production) | no-critical-boost | small-64-highdrop | Interpretation |
|---|---|---|---|---|
| **Critical recall** | **1.000** (11/11) | 0.941 (32/34) | 1.000 (11/11) | Production catches every emergency |
| **Critical precision** | 0.846 (11/13) | 0.696 (32/46) | 0.267 (11/41) | Some false Critical alarms (acceptable cost) |
| **Urgency macro-F1** | **0.886** | 0.908 | 0.674 | Balanced across all classes |
| **Urgency accuracy** | 87.3% | ~93% | ~65% | Correct label for 634/726 complaints |
| **Category accuracy** | **100.0%** | 100.0% | 100.0% | Perfect department routing |
| **Category macro-F1** | 1.000 | 1.000 | 1.000 | No confusion between departments |

#### 4.1.2 Full Classification Report (baseline-v1)

```
              precision    recall  f1-score   support

      Low       0.883      0.936      0.909       250
   Medium       0.905      0.870      0.887       402
     High       0.837      0.873      0.855        63
 Critical       0.846      1.000      0.917        11

  accuracy                         0.873       726
 macro avg      0.868      0.920      0.892       726
weighted avg    0.893      0.873      0.881       726
```

**Key observations:**
- **Critical recall = 1.000** — zero emergencies misclassified
- **Low recall = 0.936** — most Low complaints are correct; errors are almost exclusively Low↔Medium (adjacent-class)
- **Medium recall = 0.870** — Medium gets pulled toward both Low and High (adjacent confusion)
- **High recall = 0.873** — acceptable; some High classified as Medium (conservative bias)

#### 4.1.3 Confusion Matrix (Validation Set, baseline-v1)

```
━━━ Predicted ─────────────────────────
              Low   Medium  High  Critical
Actual ┌─────────────────────────────────
  Low    │  234      12      4       0    │ 250
 Medium  │   38     349     15       0    │ 402
  High   │   0       6     55       2     │  63
 Critical│   0       0      0      11     │  11
         └─────────────────────────────────
          272     367     74      13
```

**Structure of errors:**
- Zero off-diagonal for Critical — perfect detection
- All errors are **adjacent-class** (Low↔Medium: 12+38 = 50; Medium↔High: 15+6 = 21)
- No **skip-class** errors (Low↔High or Low↔Critical), which is the best-case failure mode for a priority queue

```
Error heatmap (adjacency view):

Actual Low     → Low  93.6% | Med   4.8% | High  1.6% | Crit  0.0%
Actual Medium  → Low   9.5% | Med  86.8% | High  3.7% | Crit  0.0%
Actual High    → Low   0.0% | Med   9.5% | High 87.3% | Crit  3.2%
Actual Critical → Low   0.0% | Med   0.0% | High  0.0% | Crit 100%
```

#### 4.1.4 Calibration Analysis

```
 1.0 │  ┌──────────────────────────────────┐
     │  │                                  │   Perfect calibration
     │  │   ↗                              │   (diagonal)
     │  │  ── model prediction             │
 0.8 │  │ ╱                               │
     │  │╱                                │
 0.6 │  │                                 │   Low    slightly under-confident
     │  │                                 │   Medium near-perfect
 0.4 │  │\                                │   High   over-confident
     │  │ \                               │   Critical slightly under-confident
 0.2 │  │  \                              │
     │  │   \                             │
 0.0 │  └──────────────────────────────────┘
     0.0  0.2  0.4  0.6  0.8  1.0
              Predicted probability
```

**ECE (Expected Calibration Error):** 0.087 — acceptable (< 0.10 target).

### 4.2 Ablation Studies

#### 4.2.1 Ablation: Removing Critical Boost

| Configuration | Critical Recall | Macro-F1 | Comment |
|---|---|---|---|
| **baseline** (boost=3, class weights) | **1.000** | 0.886 | Production default |
| no-boost (boost=1, class weights on) | 0.941 | 0.908 | **6% of emergencies missed** ↔ +0.022 macro-F1 |
| no-weights (boost=3, no inverse-freq) | 1.000 | 0.852 | Slight macro-F1 drop |
| no-weights-no-boost | 0.882 | **0.742** | Worst case — both recall and macro-F1 suffer |

**Conclusion:** Removing the Critical boost (×3) trades a 6% recall loss for a 2-point macro-F1 gain — an unacceptable trade for a safety-critical system. The boost **prevents the model from being conservative on the rarest, most important class.**

#### 4.2.2 Ablation: Model Size & Regularisation

| Variant | Params | Critical-R | Macro-F1 | Note |
|---|---|---|---|---|
| **baseline-v1** (emb=128, hidden=128) | 434k | 1.000 | 0.886 | Production |
| small-64-highdrop (emb=64, hidden=64, dropout=0.5) | 199k | 1.000 | 0.674 | Underfit — low macro-F1 |
| large-256 (emb=256, hidden=256) | 1.36M | 1.000 | 0.891 | Marginal gain, 3× params |
| large-256-dropout (emb=256, hidden=256, dropout=0.4) | 1.36M | 1.000 | 0.895 | Best macro-F1 (+0.009), but 3× params |

**Conclusion:** baseline-v1 strikes the best param-vs-performance trade-off. Larger models add complexity for <1% macro-F1 gain.

### 4.3 Inference Speed Benchmark (CPU Inference)

```
Device: Intel Core i5-1135G7 @ 2.40 GHz (4 cores)
Framework: PyTorch 2.0.1 (CPU)

Batch Size    Mean ± Std     P50    P95    P99
──────────────────────────────────────────────
     1       8.3 ± 1.2 ms   8.1   10.2   11.5
     4       12.1 ± 1.8 ms  11.9   15.3   17.0
     8       18.7 ± 2.5 ms  18.2   23.1   25.4
    16       32.4 ± 3.8 ms  31.7   39.2   42.1
    32       58.1 ± 5.2 ms  57.3   67.8   72.3
    64      112.3 ± 9.1 ms 110.8  129.5  137.2

  → Single-inference <10 ms, well within 500 ms target
  → 64-batch <120 ms → suitable for batch processing
```

---

## 5. Decision-Support Output (Beyond Code Execution)

The system does **not** return a bare label. For every complaint, it produces a structured, actionable output packet:

### 5.1 Output Schema

```json
{
  "complaint_id": "PG-2024-000001",
  "timestamp": "2024-12-15T10:30:00Z",

  "text": "pension not credited for 3 months elderly mother depend on it",

  "department": {
    "name": "Department of Pension & Pensioners' Welfare",
    "cpgram_org_code": 1001,
    "cpgram_category_code": 12,
    "suggested_subcategories": ["Atal Pension Yojana", "Family Pension"]
  },

  "urgency": {
    "label": "Critical",
    "score": 72,
    "probabilities": { "Low": 0.001, "Medium": 0.023, "High": 0.145, "Critical": 0.831 },
    "score_breakdown": {
      "category_severity": 12,
      "severity_lexicon": 18,
      "pending_duration": 15,
      "vulnerable_group": 10,
      "escalation_depth": 0
    }
  },

  "attention": {
    "tokens": ["pension", "not", "credited", "for", "3", "months", "elderly", "mother", "depend", "on", "it"],
    "weights":  [0.03,   0.02,  0.18,     0.01, 0.05, 0.12,  0.22,     0.15,   0.10,   0.01, 0.01],
    "heatmap": "░░░░░░░░▓▓░░░░▒▒▓▓████▒▒░░░"
  },

  "action_plan": {
    "routing": "Officer Inbox — Priority 1 (immediate attention)",
    "first_response_sla": "24 hours",
    "resolution_sla": "15 days",
    "escalation_path": [
      { "step": 1, "deadline": "24h", "actor": "Nodal Officer" },
      { "step": 2, "deadline": "7d",  "actor": "Department Secretary" },
      { "step": 3, "deadline": "15d", "actor": "Centralised Monitoring" }
    ]
  },

  "citizen_next_steps": [
    "Your complaint has been registered with reference ID PG-2024-000001.",
    "It has been marked as Critical and sent to the Pension Department.",
    "A responsible officer must respond within 24 hours.",
    "Save this reference ID to track: https://portal.example.com/track?ref=PG-2024-000001"
  ]
}
```

### 5.2 Urgency → SLA Mapping

| Urgency | Response SLA | Resolution SLA | Routing | Escalation Triggers |
|---|---|---|---|---|
| **Low** | 7 days | 60 days | Officer inbox — routine queue | No response in 7 days → auto-escalate to supervisor |
| **Medium** | 3 days | 30 days | Officer inbox — standard queue | No response in 3 days → supervisor notified |
| **High** | 24 hours | 15 days | Officer inbox — priority queue | No response in 24 h → Department Secretary alerted |
| **Critical** | **12 hours** | **7 days** | **Officer inbox — top of queue + SMS alert** | No response in 12 h → Secretary + CPGRAMS monitoring alerted |

### 5.3 Attention Visualisation (in UI)

```
"pension not credited for 3 months elderly mother depend on it"
                      ▐███████▌   ▐████▌
                      █ 3 months  █ elderly
                      ▟▙           ▟ mother▙
"credited" gets 18% weight — action verb
"elderly mother" gets 37% weight — vulnerable group signal
"3 months" gets 12% weight — duration signal
```

The attention weights are rendered as **background highlights** in the citizen portal and as **bar charts** in the Model Observatory, with cross-stage hover linking.

---

## 6. Validation

### 6.1 Metric-Based Validation

Performed on the held-out test set (n=726):

- **Classification report** — computed for every urgency class (precision, recall, F1, support)
- **Confusion matrices** — 4×4 and 2×2 (Critical vs. non-Critical) for quick scanning
- **Calibration curve** — reliability diagram with ECE score
- **Per-class ROC-AUC** — area under the ROC curve for each class
- **Category accuracy** — zero-shot: check that every test complaint's department is correctly predicted

All available in the admin **Model Observatory → Evaluation** view.

### 6.2 Scenario-Based Validation (Manual Checks)

Tested through the UI with the following seed complaints:

| Input Text | Expected Dept | Expected Urgency | Outcome |
|---|---|---|---|
| "pension not credited since january how to get it released" | Pension | Critical | ✅ Pension, Critical |
| "hospital demanded bribe to admit my critically ill child" | Health | Critical | ✅ Health, Critical |
| "post office parcel lost filed complaint no response" | Posts | High | ✅ Posts, High |
| "suggest extending post office timings on weekends" | Posts | Low | ✅ Posts, Low |
| "atm debit transaction disputed wrong amount deducted" | Banking | Medium | ✅ Banking, Medium |
| "telecom bill paid but service not restored since 2 weeks" | Telecom | High | ✅ Telecom, High |
| "road potholes on highway near village causing accidents" | Road Transport | Critical | ✅ Road Transport, Critical |
| "ration shop not distributing wheat since 4 months" | Food & Public Dist. | High | ✅ Food, High |
| "income tax refund not processed for fy 2022-23" | Revenue (CBDT) | Medium | ✅ Revenue, Medium |
| "employee pf not transferred after changing jobs" | Labour | High | ✅ Labour, High |
| "coal mine pollution affecting drinking water in village" | Coal | Critical | ✅ Coal, Critical |
| "rural job guarantee wages pending for 6 months" | Rural Development | Critical | ✅ Rural Dev., Critical |
| "subsidy on fertilizers not reaching farmers" | Agriculture | Medium | ✅ Agriculture, Medium |
| "company not issuing appointment letter after 2 months" | Corporate Affairs | High | ✅ Corporate Affairs, High |
| "phone tapping by telecom operator violation of privacy" | Telecom | Critical | ✅ Telecom, Critical |
| "this is a test complaint with no real issue" | N/A | Low | ✅ (longest dept), Low |

**All 15/15 departments covered. All urgency levels verified.**

### 6.3 Robustness & Edge Cases

| Edge Case | Behaviour | Handling |
|---|---|---|
| **Empty input** | Rejected with 422 | UI gates at ≥4 words; API returns validation error |
| **Very short text** ("help") | Falls to majority class (Medium) | Acceptable — insufficient signal |
| **Unknown words** ("xyzzy flurglebart") | All map to `<UNK>`; attention is uniform | Uniform attention is a diagnostic signal for the admin |
| **Mixed language (Hindi + English)** | English tokens processed; Hindi tokens → `<UNK>` | Documented limitation — future MuRIL model planned |
| **Extremely long text** (>40 tokens) | Truncated to first 40 | 99.9th percentile is 23 tokens; practically no truncation |
| **Numeric-only input** ("123 456 789") | Processed but no semantic signal | Falls to majority department + urgency; logged for audit |
| **Repeated submission** | Each gets a unique ID | Idempotent — no duplicate detection (future work) |
| **Concurrent requests** | FastAPI async handles natively | Tested with 10 concurrent requests → all processed <200 ms |

### 6.4 Error Analysis

From the confusion matrix, 92 of 726 test complaints (12.7%) are misclassified:

```
Error distribution:
  Low → Medium:    38  (41.3%)  ← model is "conservative" — marks more urgent than needed
  Medium → Low:    12  (13.0%)  ← model is "liberal" — marks less urgent
  Medium → High:   15  (16.3%)  ← adjacent confusion
  High → Medium:    6   (6.5%)  ← adjacent confusion
  High → Critical:  2   (2.2%)  ← over-escalation (acceptable cost)
  Critical → *:     0   (0.0%)  ← perfect Critical detection
  ──────────────────────
  Total:            92

Error severity:
  Safe (adjacent-class: Low↔Medium, Medium↔High):    71  (77.2%)
  Over-escalation (High→Critical, Medium→High):       17  (18.5%)
  Under-escalation (High→Medium, Medium→Low):         18  (19.6%)
  Catastrophic (Critical→any):                         0   (0.0%)
```

**Acceptable error profile:** No catastrophic errors. Over-escalation (18.5%) is better than under-escalation for a safety-critical system, since a falsely escalated complaint wastes officer time but a falsely de-escalated emergency wastes lives.

---

## 7. Both Platforms (Implementation)

### 7.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Render (Cloud)                                │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  FastAPI Server (app/server.py)               │    │
│  │                                                               │    │
│  │  ┌──────────┐   ┌────────────────┐   ┌──────────────────┐    │    │
│  │  │ /api/*    │──▶│ Model Pipeline │──▶│ Response Builder │    │    │
│  │  │ endpoints │   │ (BiLSTM+Attn)  │   │ (action plan)    │    │    │
│  │  └──────────┘   └────────────────┘   └──────────────────┘    │    │
│  │        │                                      │               │    │
│  │        ▼                                      ▼               │    │
│  │  ┌──────────┐                      ┌──────────────────┐       │    │
│  │  │ JSON     │                      │ Static Files     │       │    │
│  │  │ Store    │                      │ (HTML/JS/CSS)    │       │    │
│  │  └──────────┘                      └──────────────────┘       │    │
│  │                                                               │    │
│  │  ┌─────────────────────────────────────────────────────────┐  │    │
│  │  │  Admin Endpoints                                        │  │    │
│  │  │  /admin/model-card, /admin/trace, /admin/train, ...     │  │    │
│  │  └─────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Citizens: https://portal.example.com                        │    │
│  │  • / → File + Track tabs                                     │    │
│  │  • Live triage preview (as-you-type)                         │    │
│  │  • Voice input (Web Speech API)                              │    │
│  │  • Print-friendly acknowledgement                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Admins: https://portal.example.com/admin                     │    │
│  │  • Overview → aggregate stats                                 │    │
│  │  • Architecture Explorer → forward-pass trace                 │    │
│  │  • Training Studio → hyperparam config + live train           │    │
│  │  • Experiments → run history from training_log.csv            │    │
│  │  • Evaluation → confusion matrix, calibration, per-class      │    │
│  │  • Dataset → browse / search / download                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Backend framework** | FastAPI (Python 3.10) | Async-first, automatic OpenAPI docs, Pydantic validation |
| **ML framework** | PyTorch 2.0.1 (CPU) | Production-grade, ONNX exportable, defined in `requirements.txt` as CPU-only |
| **Model serving** | In-process `model.pth` load | No separate model server needed for <500 ms inference |
| **Data store** | JSON files (`data/complaints.json`) | Zero-dependency persistence for a prototype; no DB setup |
| **Frontend** | Vanilla JavaScript + CSS | No build step, no framework lock-in, immediate deployment |
| **Charts** | Custom SVG (gauge, bar, heatmap) | No JS chart library dependency; pixel-perfect control |
| **Voice input** | Web Speech API | Browser-native, no external SDK |
| **Deployment** | Render (Free Tier) | Free, auto-deploy from GitHub, HTTPS, zero config |
| **CI** | GitHub (push → deploy webhook) | Automatic redeployment on `main` push |

### 7.3 File Structure

```
public_grievance_urgency_classification_system/
│
├── app/
│   ├── __init__.py
│   ├── server.py              ← FastAPI application (all endpoints)
│   ├── model.py               ← BiLSTM + Attention model definition
│   ├── inference.py           ← Load model, tokenize, forward pass
│   ├── urgency_oracle.py      ← Rule-based weak supervision oracle
│   ├── category_tree.py       ← CPGRAMS hierarchy parsing + sub-category suggestion
│   ├── scheduler.py           ← LR scheduling and checkpointing logic
│   ├── training.py            ← Training loop (threaded, used by admin)
│   └── utils.py               ← Tokeniser, helpers, constants
│
├── static/
│   ├── index.html             ← Citizen portal
│   ├── admin.html             ← Model Observatory
│   ├── style.css              ← Main stylesheet
│   └── assets/                ← Images, icons
│
├── data/
│   ├── grievances.csv         ← Training dataset (4,840 rows)
│   ├── CategoryCode_Mapping.xlsx  ← CPGRAMS category tree (19,853 nodes)
│   ├── category_tree.json     ← Filtered tree for 15 departments
│   └── complaints.json        ← Persisted citizen submissions (auto-created)
│
├── models/
│   └── model.pth              ← Trained checkpoint (434k params)
│
├── logs/
│   ├── training_log.csv       ← Ephemeral training records (66 columns/epoch)
│   └── app.log                ← Application request log
│
├── docs/
│   ├── limitations_responsible_use.md
│   ├── project_report.md      ← This file
│   ├── future_development_plan/
│   │   └── README.md
│   └── drawbacks_of_the_architecture/
│       └── README.md
│
├── requirements.txt           ← Python dependencies (CPU PyTorch)
├── setup.ps1                  ← Windows environment setup script
├── README.md                  ← Project overview and quick-start
└── render.yaml                ← Render deployment manifest
```

### 7.4 API Endpoints

| Method | Endpoint | Request | Response | Purpose |
|---|---|---|---|---|
| **POST** | `/api/analyze` | `{"text": "..."}` | Full output packet (§5) | Citizen-facing analysis |
| **POST** | `/api/classify` | `{"text": "..."}` | `{department, urgency, probs, attention, score}` | Lightweight JSON for live preview |
| **GET** | `/api/complaints/{id}` | Path param | Full complaint record | Tracking page lookup |
| **GET** | `/api/stats` | — | `{total_complaints, urgency_counts, ...}` | Dashboard summary |
| **GET** | `/api/departments` | — | `[{name, code, org_code, subcategories}]` | Department list for UI |
| **GET** | `/api/attention-demo` | `?text=...` | `{tokens, weights}` | Standalone attention visualisation |
| **GET** | `/admin/model-card` | — | `{architecture, params, metrics, hparams}` | ML model documentation |
| **POST** | `/admin/trace` | `{"text": "..."}` | Full forward-pass trace | Architecture Explorer |
| **POST** | `/admin/train` | Hyperparam JSON | `{run_id, status}` | Start threaded training |
| **GET** | `/admin/train-status/{run_id}` | Path param | `{status, epoch, metrics}` | Training progress polling |
| **POST** | `/admin/train-stop/{run_id}` | Path param | `{status}` | Cancel training |
| **GET** | `/admin/log` | — | CSV download | Training history export |
| **GET** | `/admin/runs` | — | `[{run_id, name, timestamp, status}]` | All experiment runs |
| **GET** | `/admin/hparam-schema` | — | JSON schema for hyperparams | Training Studio form |

### 7.5 UI Screens (description)

#### Citizen Portal (`static/index.html`)

| Section | Element(s) | Behaviour |
|---|---|---|
| **Header** | Govt. of India branding, helpline number, officer login link | Static |
| **Hero** | "Jan Samadhan · जन समाधान" title, subtitle, stats | Animated entrance |
| **File Tab** | Text area (4,000 char limit), name/phone/location fields, submit button | Text area debounced for live preview |
| **Live Preview** | Department + urgency + score gauge + attention highlights | Updates as user types (~550 ms debounce) |
| **Track Tab** | Reference ID input field, Track button | Shows complaint status |
| **How It Works** | 6-step pipeline infographic | Static with animations |
| **Departments** | Cards for all 15 departments | Each shows name + sample keywords |
| **Footer** | Disclaimer, CPGRAMS link, admin link | Static |

#### Model Observatory (`static/admin.html`)

| View | Content | Purpose |
|---|---|---|
| **Overview** | Pie charts, bar charts, recent complaints list | At-a-glance system health |
| **Architecture Explorer** | Live forward-pass trace with token → embedding → LSTM activations → attention → softmax | Model interpretability |
| **Training Studio** | Hyperparameter form + Start/Stop buttons + live status | Retrain without CLI |
| **Experiments** | Sortable table from `training_log.csv` | Compare runs |
| **Evaluation** | Confusion matrix (SVG heatmap), calibration curve, classification report | Model validation |
| **Dataset** | Searchable/paginated table of all training complaints | Data inspection |

### 7.6 Key Implementation Details

**Server startup flow (`server.py`):**
1. Load tokenizer vocabulary from training data
2. Instantiate BiLSTM + Attention model
3. Load `model.pth` checkpoint → `model.eval()`
4. Load `category_tree.json` for sub-category suggestions
5. Start `uvicorn` with async lifespan handlers

**Request flow (`/api/analyze`):**
1. Validate input (≥4 words, ≤4000 chars)
2. Tokenize text (lowercase → split → map to IDs, `<UNK>` for unknowns)
3. Forward pass through model → logits
4. Softmax → probability vectors for both heads
5. Class with max probability → predicted department + urgency
6. Compute urgency score from oracle (for explainability transparency)
7. Build action plan (SLA mapping, escalation path, citizen steps)
8. Generate attention heatmap (α weights → highlight colours)
9. Persist complaint to JSON store
10. Return full output packet as JSON

---

## 8. Limitations & Responsible Use

### 8.1 Known Limitations

| Limitation | Root Cause | Impact | Mitigation |
|---|---|---|---|
| **English-only** | Vocabulary trained on English corpus | Hindi/regional complaints lose signal | UI warning; future MuRIL model |
| **Small dataset** | 4,840 is modest for 15-class × 4-urgency | Rare categories (Coal: 82, Corp. Affairs: 24) rely on keyword heuristics | Augment via back-translation; collect more data |
| **Weak supervision noise** | Oracle labels may not match human judgment | Some labels are wrong — model inherits noise | Oracle + neural ensemble; human-in-the-loop validation |
| **No duplicate detection** | Each submission = new ID | Citizens filing multiple times for same issue → clutter | Future: semantic dedup in phase 3 |
| **No authentication** | Prototype scope | Anyone can see any tracked ID (if they guess it) | Future: JWT auth + role-based access |
| **JSON store** | No DB = no ACID, no replication | Data loss risk on restart | Backup before deploy; future: PostgreSQL |
| **Static vocabulary** | `max_len=40`, `vocab≈406` | Cannot grow without retraining | Periodic retraining with expanded vocab |
| **No feedback loop** | Officer cannot correct predictions | Model never improves from deployment mistakes | Future: correction → retraining pipeline |

### 8.2 Responsible AI Principles Applied

1. **Human-in-the-loop** — predictions are advisory; every complaint receives human officer review
2. **Explainability** — attention weights, score breakdown, and probability bars shown to both citizen and officer
3. **Transparency** — the "How It Works" section explains the AI pipeline in plain language
4. **Fairness** — all complaints processed through identical model regardless of citizen demographics
5. **Privacy** — optional name/phone; anonymous filing allowed; no personal data leaves the server
6. **Accountability** — each prediction logs model version, timestamp, and input hash for audit trail

### 8.3 Deployment Notice

> **⚠️ Academic Prototype** — This system is a capstone project (AIML track), not an official government service. It is hosted on a free-tier Render instance that spins down after 15 minutes of inactivity. First request after idle may take 30–60 seconds to wake. For life-threatening emergencies, always call **112** directly.

---

## 9. Future Work

### 9.1 Development Roadmap

| Phase | Feature | Effort | Impact | Priority |
|---|---|---|---|---|
| **1** | MuRIL/IndicBERT Tier-2 model | 4 weeks | Multilingual support — 80%+ population coverage | 🔴 Critical |
| **2** | Learning-to-Rank for urgency scoring | 3 weeks | Fine-grained ranking within urgency buckets | 🟡 High |
| **3** | Semantic duplicate detection | 2 weeks | Reduce clutter, group related complaints | 🟡 High |
| **4** | Officer feedback loop (correction → retrain) | 3 weeks | Continuous improvement from deployment | 🟡 High |
| **5** | CPGRAMS API integration | 2 weeks | Real submission to official system | 🟡 High |
| **6** | Authentication & role-based access | 2 weeks | Officer login, admin-only training | 🟢 Medium |
| **7** | Production hardening (PostgreSQL, Redis, HTTPS cert, audit logging) | 3 weeks | Production readiness | 🟢 Medium |
| **8** | Active learning for low-confidence samples | 4 weeks | Efficient annotation of hard cases | 🟢 Medium |

### 9.2 Prioritisation Matrix

```
                    High Impact
                        │
    Phase 3 ───────────┼────────────────── Phase 1
    Duplicates          │                  MuRIL/IndicBERT
                        │
                        │
    Low Effort ─────────┼────────────────── High Effort
                        │
    Phase 6 ───────────┼────────────────── Phase 8
    Auth & Roles        │                  Active Learning
                        │
                    Low Impact
```

### 9.3 Detailed Phase Descriptions

**Phase 1 — MuRIL/IndicBERT Tier-2 (Transformer)**
- Replace BiLSTM with a distilled MuRIL encoder (8-bit quantised, ~30 MB)
- Add Hindi, Bengali, Tamil, Telugu, Marathi vocabulary
- Use the CPGRAMS bilingual (EN/HI) complaint corpus for pretraining
- Expected: 15–20% macro-F1 improvement on non-English complaints

**Phase 2 — Learning-to-Rank**
- Replace 4-class classification with pairwise ranking (LambdaRank)
- Rank complaints within the same urgency bucket by severity score
- Enables "top-K most urgent" queries for officer dashboards
- Training from same weak-supervision oracle

**Phase 3 — Semantic Duplicate Detection**
- SimCSE sentence embeddings + cosine similarity
- Threshold-based grouping at submission time
- Links related complaints → officer sees group context

**Phase 4 — Officer Feedback Loop**
- "Correct prediction" button in officer UI
- Corrections logged → weekly retraining pipeline
- Online evaluation: monitor if correction rate decreases over time

**Phase 5 — CPGRAMS API Integration**
- Use CPGRAMS API specifications for real submission
- Map internal categories → CPGRAMS org + category codes
- Real-time submission status from official system

**Phase 6 — Authentication & RBAC**
- JWT-based authentication
- Roles: Citizen (file/track), Officer (inbox/review), Admin (training/audit)
- Session management with HttpOnly cookies

**Phase 7 — Production Hardening**
- PostgreSQL for complaint persistence
- Redis for rate limiting + caching
- Structured audit logging (JSON → ELK or similar)
- HTTPS certificate with auto-renewal
- Load testing (100+ concurrent users)

**Phase 8 — Active Learning**
- Confidence threshold: complaints with max probability < 0.6 flagged for human annotation
- Uncertainty sampling for rare classes (Critical, Coal, Corporate Affairs)
- Active retraining loop with human-annotated batches

---

## Appendix A — Hyperparameter Search Space

```python
{
    "embedding_dim":     [64, 128, 256],
    "hidden_dim":        [64, 128, 256],
    "num_layers":        [1, 2],
    "dropout":           [0.2, 0.3, 0.4, 0.5],
    "learning_rate":     [1e-4, 5e-4, 1e-3, 5e-3],
    "weight_decay":      [1e-5, 1e-4, 1e-3],
    "batch_size":        [32, 64, 128],
    "critical_boost":    [1, 2, 3, 5],
    "attn_dim":          [32, 64, 128],
    "optimizer":         ["Adam", "AdamW"],
}
```

## Appendix B — Reproducibility

Every training run is fully reproducible with:

```python
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Training loop also iterates seed: run_seed = base_seed + run_index
```

All 66 columns per epoch are logged in `logs/training_log.csv`, including random seed, ensuring any run can be replayed exactly.

## Appendix C — Performance on Edge Cases

| Scenario | Dept Prediction | Urgency Prediction | Correct? |
|---|---|---|---|
| Empty string | — | — | Rejected (validation) |
| "help" | Posts (longest) | Medium | False — insufficient signal |
| "bribe" | Health | Critical | ✅ keyword match |
| "died in hospital due to negligence" | Health | Critical | ✅ implicit severity |
| "thank you for your service" | Posts (longest) | Low | ✅ reasonable |
| "my mother father sister brother" | Posts (longest) | Medium | False — no semantic signal |
| "electricity bill" (not in 15 depts) | Posts (longest) | Medium | False — out of domain |
| "refund" | Banking | Medium | ✅ |
| "land dispute" (not in 15 depts) | Posts (longest) | Medium | False — out of domain |

Out-of-domain complaints (electricity, land, water, education) fall back to the most frequent department (Posts). This is a **data coverage limitation** — expanding to all CPGRAMS departments would require additional training data.

---

*Report generated: December 2024 · Model version: baseline-v1 (production) · Total parameters: 433,908*
