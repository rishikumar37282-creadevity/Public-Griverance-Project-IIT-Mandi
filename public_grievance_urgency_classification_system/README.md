# Jan Samadhan — Public Grievance Urgency Classification System

**Track:** AIML · **Domain:** Public service and civic operations · **Difficulty:** Intermediate

A complete, working system that reads a citizen's complaint in plain English and instantly
predicts **(a) the responsible department/category**, **(b) the urgency level (Low / Medium /
High / Critical)** with an explainable 0–100 urgency score, and **(c) a concrete escalation and
action plan** — so serious issues reach officers faster instead of waiting in a first-come queue.

It ships as **two full web applications** on one FastAPI backend:

| Platform | URL | For | What it does |
|---|---|---|---|
| **Jan Samadhan citizen portal** | `http://localhost:8017/` | Citizens / users | File complaints with a **live AI triage preview while typing**, voice input, attention-based "why" explanation, urgency gauge, CPGRAMS sub-category suggestions, action plan with SLAs, reference-ID tracking, printable acknowledgement |
| **Model Observatory admin app** | `http://localhost:8017/admin` | Officers / ML admins | Model card & parameter breakdown, **interactive layer-by-layer architecture explorer**, **"Magic of the Architecture"** guided animated tour with training replay, **Training Studio** with every hyperparameter as a live control and per-epoch loss/accuracy/F1/Critical-recall curves + live confusion matrix, **experiment comparison** from a persistent `training_log.csv`, full test-set evaluation, dataset explorer |
| **Live Model Explainer** | `http://localhost:8017/explainer` | Everyone | The **whole network as one flowing canvas** (inspired by the [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)): live inputs, real activations at every layer, attention ribbons whose width = weight, hover-to-trace tokens, guided layer walk |
| **Project Story deck** | `http://localhost:8017/story` | Reviewers / presentations | **Animated web slideshow** documenting the whole project phase by phase — problem, data, architecture rationale, loss design, training, monitoring, tuning ablations, results, product, limitations, learnings — with charts drawn from the real training logs |

---

## 1. Problem statement

Government and institutional grievance systems receive thousands of complaints. Manual triage
means a pensioner whose only income stopped eight months ago waits in the same queue as a
suggestion to add counters at a post office. This project builds a text-classification system
that identifies complaint **category** and **urgency** so serious issues are handled faster.

## 2. Dataset / reference source

- **Government of India Grievance Report** (Kaggle:
  <https://www.kaggle.com/datasets/ayushyajnik/government-of-india-grievance-report>)
- [`data/grievances.csv`](data/grievances.csv) — 4,840 complaints across **15 categories/departments**
  with CPGRAMS category & organisation codes and weak-supervision urgency labels
  (Low 34.6% / Medium 38.7% / High 24.3% / **Critical 2.4%** — heavily imbalanced, by design).
- [`data/CategoryCode_Mapping.xlsx`](data/CategoryCode_Mapping.xlsx) — the official CPGRAMS category
  tree (19,853 nodes). The 3,899 nodes belonging to our 15 organisations are extracted to
  `data/category_tree.json` and used to suggest the most specific sub-category for each complaint.

## 3. AI / innovation component

1. **BiLSTM + additive attention** network (PyTorch, ~435k parameters) with **two parallel MLP
   heads** trained jointly: Head A → category (15 classes), Head B → urgency (4 classes).
2. **Class-weighted multi-task loss with a `critical_boost`** multiplier that protects the rare
   Critical class — the headline metric is **Critical-class recall**, never plain accuracy.
3. **Weak-supervision urgency oracle** — an explainable rule engine scoring severity keywords,
   pending durations ("for 8 months"), vulnerable groups (elderly, children, pregnant), and
   escalation depth into a 0–100 score shown to the citizen.
4. **Built-in explainability** — the attention weights are surfaced in both apps: citizens see
   which words drove the decision; admins see the full forward-pass trace layer by layer.
5. **Escalation recommendation engine** — urgency maps to routing level, response SLAs, an
   officer-side escalation path and citizen next steps.

## 4. How to run

```bash
cd public_grievance_urgency_classification_system
python -m venv .venv && .venv\Scripts\activate        # Windows (use source .venv/bin/activate on Linux/Mac)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# (already included) rebuild the CPGRAMS tree + train the production model:
python scripts/build_category_tree.py
python -m src.main train --run-name baseline-v1

# start both apps:
uvicorn app.server:app --port 8017
# → citizen portal: http://localhost:8017   · admin: http://localhost:8017/admin
```

CLI extras: `python -m src.main predict "my pension stopped 8 months ago"` ·
`python -m src.main evaluate`

## 5. Project workflow

```
grievances.csv ──► preprocessing (tokenise → vocab → pad to max_len)
                └► urgency oracle (weak labels + 0–100 score)
      70/15/15 stratified split ──► BiLSTM+Attention, 2 heads
      class-weighted multi-task CE (critical_boost ×3) ──► early stop on Critical-recall
      per-epoch metrics + confusion matrix ──► logs/training_log.csv
      best checkpoint ──► models/bilstm_attn.pt + bilstm_meta.json
      FastAPI ──► citizen portal + Model Observatory
```

## 6. Results (held-out 15% test split, 726 complaints)

| Metric | Value |
|---|---|
| **Critical-class recall (headline)** | **1.000** |
| Urgency macro-F1 | 0.886 |
| Urgency accuracy | 87.3% |
| Category (department) accuracy | 100.0% |

Ablation logged in `logs/training_log.csv`: removing class weights & the critical boost
(`no-critical-boost` run) raises macro-F1 slightly (0.908) but **drops Critical-recall to 0.941 —
i.e. ~6% of emergencies missed**, which is exactly the trade-off the loss design prevents.

## 7. Repository structure

```
public_grievance_urgency_classification_system/
├── data/                # grievances.csv, CategoryCode_Mapping.xlsx, category_tree.json
├── notebooks/           # BiLSTM_Attention_Training.ipynb (original end-to-end notebook)
├── src/                 # config, preprocessing, urgency_oracle, model, train, inference,
│                        # escalation, data_utils, main (CLI)
├── app/
│   ├── server.py        # FastAPI backend (citizen + admin APIs, live training)
│   └── static/          # index.html (portal), admin.html (observatory), css/, js/
├── models/              # trained artifacts (bilstm_attn.pt, bilstm_meta.json)
├── logs/                # training_log.csv — every epoch of every run ever trained
├── scripts/             # build_category_tree.py
├── docs/                # project_report.md, presentation_outline.md, limitations_responsible_use.md
├── requirements.txt
└── README.md
```

## 8. Demo screenshots

Run the app and see: citizen portal live-preview + result page, and the admin Architecture
Explorer / Training Studio / Experiments views (screenshots for the report can be captured
directly from `http://localhost:8017` and `/admin`).

## 9. Limitations & responsible use

See [`docs/limitations_responsible_use.md`](docs/limitations_responsible_use.md). Key points:
English-only training data (~4.8k rows, 15 departments), advisory predictions with mandatory
human review, illustrative SLAs, no personal data leaves the local server, urgency labels are
weak-supervision (rule-derived), and the system must never gate access to emergency services —
call 112 for life-threatening situations.

## 10. Future improvements

- Tier-2 model: pretrained multilingual transformer (MuRIL) for Hindi/regional languages
- Learning-to-rank queue position instead of 4 discrete classes; duplicate-complaint detection
- Officer feedback loop (accept/override) as continual training signal
- Real CPGRAMS API integration and authenticated officer accounts

## 11. Team

Rishi Kumar (add teammates here).
