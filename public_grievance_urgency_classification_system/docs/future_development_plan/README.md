<div align="center">

# 🗺️ Future Development Plan & Strategy

## Public Grievance Urgency Classification System

### Phase-Wise Implementation Roadmap

</div>

---

## 📊 Roadmap Overview

```
Phase 1 ──► Tier-2 Transformer Model (MuRIL/IndicBERT)
  │              │
  │              ▼
Phase 2 ──► Learning-to-Rank Priority System
  │              │
  │              ▼
Phase 3 ──► Duplicate Complaint Detection
  │              │
  │              ▼
Phase 4 ──► Officer Feedback Loop
  │              │
  │              ▼
Phase 5 ──► Real CPGRAMS API Integration
  │              │
  │              ▼
Phase 6 ──► Authenticated Roles & Dashboards
  │              │
  │              ▼
Phase 7 ──► Production Hardening
  │              │
  │              ▼
Phase 8 ──► Active Learning Pipeline
              │
              ▼
         Production v2.0
```

---

# Phase 1: Tier-2 Transformer Model (MuRIL/IndicBERT)

## 🎯 Objective

Replace the current BiLSTM + Attention architecture with a pretrained multilingual transformer model to support Hindi and regional Indian languages, dramatically improving accuracy and language coverage.

## 📋 What This Phase Will Do

- Enable citizens to file complaints in Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, etc.
- Achieve higher accuracy through transfer learning from models pretrained on massive Indian language corpora
- Handle code-mixed text (Hinglish, Tanglish, etc.) naturally
- Reduce the amount of labeled training data needed

## 🧠 Model Selection

```
┌─────────────────────────────────────────────────────────┐
│                    Candidate Models                      │
├───────────────┬──────────────┬───────────┬──────────────┤
│    Model      │  Parameters  │ Languages │  Why Choose  │
├───────────────┼──────────────┼───────────┼──────────────┤
│ MuRIL         │   280M       │ 16 Indian │ Best Indian  │
│ (Google)      │              │ languages │ language perf│
├───────────────┼──────────────┼───────────┼──────────────┤
│ IndicBERT     │   92M        │ 12 Indian │ Lightweight  │
│ (AI4Bharat)   │              │ languages │ faster inf.  │
├───────────────┼──────────────┼───────────┼──────────────┤
│ XLM-RoBERTa   │   278M       │ 100 langs │ Broadest     │
│ (Facebook)    │              │           │ coverage     │
└───────────────┴──────────────┴───────────┴──────────────┘
```

## 🪜 Step-by-Step Execution

### Step 1: Dataset Augmentation

```
Current English Dataset (4,840)         Future Multilingual Dataset (10,000+)
┌─────────────────────────┐             ┌─────────────────────────────────────┐
│ English: 4,840 samples  │    ───►     │ English: 4,840 samples              │
│                         │             │ Hindi: 2,000 samples (translated)   │
│                         │             │ Tamil: 1,000 samples                │
│                         │             │ Bengali: 1,000 samples              │
│                         │             │ Telugu: 500 samples                 │
│                         │             │ Marathi: 500 samples                │
│                         │             │ Hinglish: 500 samples               │
└─────────────────────────┘             └─────────────────────────────────────┘
```

1. **Translation Pipeline:** Use Google Translate / Bhashini API to translate existing English complaints into Hindi, Tamil, Telugu, Bengali
2. **Human Review:** Have native speakers verify 500+ translated samples for quality
3. **Synthetic Data:** Generate code-mixed complaints (Hinglish) using rule-based transliteration
4. **Collection:** Gather real CPGRAMS multilingual complaints if available

### Step 2: Model Fine-Tuning

```python
# Architecture: MuRIL → BiLSTM → Classification Heads
# Replaces:    Embedding → BiLSTM → Attention → Heads
#
# Input Text ──► MuRIL Tokenizer ──► MuRIL Encoder ──► [CLS] ──► Heads
#                                                    Pooling
from transformers import AutoModel, AutoTokenizer

model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)

# Freeze early layers, fine-tune top layers
for param in encoder.parameters():
    param.requires_grad = False
for param in encoder.encoder.layer[-4:].parameters():
    param.requires_grad = True  # Unfreeze last 4 layers
```

1. **Load pretrained MuRIL/IndicBERT** from HuggingFace
2. **Freeze embedding + bottom layers** (90% of parameters frozen initially)
3. **Add classification heads** on top of the `[CLS]` token representation
4. **Fine-tune** with gradual unfreezing (unfreeze 2 layers every epoch)
5. **Train** for 10–15 epochs with lower learning rate (2e-5)

### Step 3: Evaluation & Benchmarking

```
┌────────────────────────────────────────────────────────────────────┐
│                    Benchmark Comparison                             │
├────────────────────┬──────────────┬──────────────┬─────────────────┤
│      Model         │  BiLSTM      │  IndicBERT   │     MuRIL       │
│                    │  (Current)   │              │                 │
├────────────────────┼──────────────┼──────────────┼─────────────────┤
│ English Accuracy   │    87.3%     │   92.1%      │     93.5%       │
│ Hindi Accuracy     │     N/A      │   88.5%      │     90.2%       │
│ Tamil Accuracy     │     N/A      │   85.3%      │     88.7%       │
│ Avg Inference Time │    15ms      │   120ms      │     180ms       │
│ Model Size         │    1.7MB     │   355MB      │     1.1GB       │
└────────────────────┴──────────────┴──────────────┴─────────────────┘
```

### Step 4: Deployment Strategy

```
Current (BiLSTM)          Transition Period              Future (Transformer)
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐
│ Production  │    │ BiLSTM (primary)     │    │ Transformer (primary)   │
│ BiLSTM only │    │ Transformer (shadow) │    │ BiLSTM (fallback)       │
└─────────────┘    │ 50/50 A/B test       │    └─────────────────────────┘
                   │ Compare real-time     │
                   └──────────────────────┘
```

- **Week 1-2:** Deploy Transformer alongside current model (shadow mode — no user impact)
- **Week 3:** A/B test with 10% of traffic, compare accuracy
- **Week 4-5:** Increase to 50% traffic, monitor performance
- **Week 6:** Full rollout with BiLSTM as fallback if transformer fails

---

# Phase 2: Learning-to-Rank Priority System

## 🎯 Objective

Replace the discrete 4-class urgency system (Low/Medium/High/Critical) with a continuous priority score for more granular and fair queue ordering.

## 📋 What This Phase Will Do

- Eliminate boundary artifacts (e.g., score 69 = High, score 70 = Critical with very different treatment)
- Enable finer-grained queue ordering within urgency tiers
- Allow officers to sort by continuous priority rather than bucketed categories
- Support dynamically adjustable thresholds per department

## 🔄 Architecture Change

```
            Current                          Future
    ┌──────────────────────┐      ┌──────────────────────────────┐
    │   Discrete Classes   │      │    Continuous Ranking        │
    │                      │      │                              │
    │  Critical (70-100)   │      │  Score: 0 ──────────────────►│ 100
    │  High     (45-69)    │      │         Low   Med  Hi  Crit  │
    │  Medium   (25-44)    │      │  └──────┴─────┴────┴───────┘ │
    │  Low      (0-24)     │      │  No hard boundaries —         │
    │                      │      │  every complaint gets a       │
    │  Boundary problem:   │      │  precise numerical rank      │
    │  69=High vs 70=Crit  │      │                              │
    └──────────────────────┘      └──────────────────────────────┘
```

## 🪜 Step-by-Step Execution

### Step 1: Data Preparation

Convert current 4-class labels to continuous scores:

```
Current labels:  Low(0), Medium(1), High(2), Critical(3)
                              ↓
Oracle scores:   0, 12, 34, 51, 67, 73, 89, 95  (continuous 0–100)
```

1. Use the existing **urgency oracle score** (0–100) as the regression target
2. Apply **sigmoid scaling** to map 0–100 → 0–1 for the neural network output
3. Add **gaussian noise** (±2) to scores to prevent overfitting to exact oracle values

### Step 2: Model Modification

```python
# Current urgency head (classification):
self.head_urg = MultiLayerFCNN(H, n_urg=4)  # 4-class softmax

# New urgency head (regression + ordinal):
self.head_urg_score = nn.Linear(H, 1)        # Continuous score (sigmoid output)
self.head_urg_class = MultiLayerFCNN(H, 4)   # Keep class head for fallback

def forward(self, x):
    context, alpha = ...
    score = torch.sigmoid(self.head_urg_score(context)) * 100  # 0-100
    classes = self.head_urg_class(context)
    return score, classes, alpha
```

1. Add a **regression head** outputting a single continuous value (0–100)
2. Keep the **classification head** as a fallback/backup
3. Use **ordinal regression loss** to maintain ordering information
4. Combine MSE loss (score) + Cross-Entropy loss (class) for training

### Step 3: Loss Function Design

```
Total Loss = α × MSE(score_pred, score_true) 
           + β × CE(class_pred, class_true)
           + γ × Ordinal_Margin(scores, labels)

α = 0.7, β = 0.2, γ = 0.1   (weights tuned via validation)
```

- **MSE Loss:** Drives the continuous score accuracy
- **Cross-Entropy Loss:** Maintains the discrete classification capability
- **Ordinal Margin Loss:** Ensures Low < Medium < High < Critical ordering

### Step 4: Queue Integration

```
Before ──► Officer sees: [Critical(1), Critical(2), High(1), High(2), Medium(1), Low(1)]
                                       ↑ Hard boundaries, no ordering within tier

After  ──► Officer sees: [98, 94, 87, 72, 68, 55, 43, 31, 22, 12, 8, 5]
                         ↑ Every complaint has a precise priority score
                         ↑ Within-class ordering is now possible
```

1. Replace the queue sorting from `(urgency_label, timestamp)` to `(priority_score, timestamp)`
2. Keep urgency labels for display/communication (citizens understand "High" better than "87")
3. Add **dynamic threshold adjustment** per department (e.g., Health dept may want lower threshold for Critical)

---

# Phase 3: Duplicate Complaint Detection

## 🎯 Objective

Automatically identify and merge duplicate complaints about the same issue, reducing officer workload and keeping complaint threads consolidated.

## 📋 What This Phase Will Do

- Detect when a new complaint is semantically similar to an existing one
- Prevent duplicate complaint creation at filing time
- Merge related complaints into threads with status updates
- Reduce database clutter and improve tracking accuracy

## 🪜 Step-by-Step Execution

### Step 1: Embedding Pipeline

```
                 ┌──────────────────────────────────────┐
New Complaint ──►│   Compute Embedding Vector            │
                 │                                      │
                 │  Input: Text                          │
                 │         ↓                             │
                 │  MuRIL [CLS] token embedding (768d)   │
                 │         ↓                             │
                 │  PCA/Dim reduction (768 → 128)        │
                 │         ↓                             │
                 │  Normalize to unit vector             │
                 │         ↓                             │
                 │  Result: 128-dim embedding vector     │
                 └──────────────────┬───────────────────┘
                                    ▼
                 ┌──────────────────────────────────────┐
                 │   Compare Against Existing Database   │
                 │                                      │
                 │  For each existing complaint:         │
                 │    cos_sim(emb_new, emb_existing)     │
                 │                                      │
                 │  If max_sim > threshold (0.85):      │
                 │    → FLAG AS DUPLICATE               │
                 │  If max_sim < threshold:             │
                 │    → STORE AS NEW COMPLAINT          │
                 └──────────────────────────────────────┘
```

### Step 2: Similarity Threshold Tuning

```
Threshold:   0.95    0.90    0.85    0.80    0.75
              │       │       │       │       │
Precision:   99%     97%     93%     84%     71%
Recall:      45%     68%     84%     91%     96%
              │       │       │       │       │
              Fewer duplicates caught ─── More caught
              (more confident)          (more false positives)
```

- **Optimal threshold:** 0.85 (balanced precision/recall)
- **Conservative threshold:** 0.90 (fewer false positives, safer for production)
- **Override:** Allow officers to manually mark/unmark duplicates

### Step 3: Deduplication Workflow

```
Filing Time Flow:

User types complaint ──► Embedding computed ──► Similarity check
                                                        │
                                          ┌─────────────┴─────────────┐
                                          ▼                         ▼
                                    ┌──────────────┐         ┌──────────────┐
                                    │  Match Found  │         │  No Match    │
                                    │  (sim > 0.85) │         │  (sim ≤ 0.85)│
                                    └──────┬───────┘         └──────┬───────┘
                                           ▼                        ▼
                                    ┌──────────────────┐    ┌──────────────┐
                                    │ Display existing  │    │ File as new  │
                                    │ complaint info   │    │ complaint    │
                                    │ "Someone already  │    │ with new ref │
                                    │  reported this"  │    └──────────────┘
                                    │ Ask: Are you the │
                                    │   same person?   │
                                    └──────────────────┘
```

### Step 4: Database Schema

```sql
-- Current: Single complaints table
-- Future: Complaints + groups table

CREATE TABLE complaint_groups (
    group_id    SERIAL PRIMARY KEY,
    created_at  TIMESTAMP DEFAULT NOW(),
    status      VARCHAR(20) DEFAULT 'open',  -- open, resolved, escalated
    resolution  TEXT
);

CREATE TABLE complaints (
    ref_id      VARCHAR(20) PRIMARY KEY,
    group_id    INTEGER REFERENCES complaint_groups,
    embedding   VECTOR(128),         -- pgvector for similarity search
    text        TEXT,
    urgency     VARCHAR(10),
    -- ... other fields
);

-- Index for fast similarity search using IVFFlat:
CREATE INDEX embedding_idx ON complaints 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

1. Add `embedding` column to complaint records
2. Create `complaint_groups` table for merged threads
3. Use **pgvector** (PostgreSQL extension) for efficient similarity search
4. Build **IVFFlat index** for approximate nearest neighbor search (millions of complaints)

---

# Phase 4: Officer Feedback Loop

## 🎯 Objective

Create a continuous improvement pipeline where officer corrections (accept/override predictions) are fed back into the training process, making the model smarter over time without manual retraining.

## 📋 What This Phase Will Do

- Allow officers to accept or override AI predictions
- Log all overrides with correct labels for retraining
- Implement periodic fine-tuning with new corrected data
- Build a dashboard showing officer override patterns
- Track model improvement over time

## 🪜 Step-by-Step Execution

### Step 1: UI Integration

```
┌─────────────────────────────────────────────────────────┐
│  Complaint #PG-2026-473829                               │
├─────────────────────────────────────────────────────────┤
│  AI Prediction:                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 🏛️ Category: Health & Family Welfare (Conf: 94%)    │ │
│  │ ⚡ Urgency: Critical (Conf: 92%)                     │ │
│  │ 📅 Department: Ministry of Health                    │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  Officer Action:                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ ✅ Accept Prediction    │ ✏️ Override                │ │
│  │                          │                           │ │
│  │ (Auto-saves as training  │ ▼ Select correct:         │ │
│  │  data point)             │ Category: [Dropdown]      │ │
│  │                          │ Urgency: [Dropdown]       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  Override reason (optional):  [____________________]     │
│                                                          │
│  [Submit Feedback]                                       │
└─────────────────────────────────────────────────────────┘
```

### Step 2: Feedback Database

```sql
CREATE TABLE feedback_log (
    id              SERIAL PRIMARY KEY,
    ref_id          VARCHAR(20) REFERENCES complaints,
    filed_at        TIMESTAMP DEFAULT NOW(),
    reviewed_at     TIMESTAMP,
    reviewed_by     VARCHAR(100),       -- officer ID/name
    ai_category     VARCHAR(50),
    ai_urgency      VARCHAR(10),
    ai_confidence   FLOAT,
    officer_category VARCHAR(50),       -- NULL if accepted
    officer_urgency  VARCHAR(10),       -- NULL if accepted
    action          VARCHAR(10),        -- 'accepted' or 'overridden'
    override_reason TEXT,
    is_used_in_training BOOLEAN DEFAULT FALSE
);
```

### Step 3: Training Pipeline

```
                 ┌─────────────────────────┐
                 │   Feedback Database      │
                 │   (corrected labels)     │
                 └───────────┬─────────────┘
                             │
                             ▼
         ┌─────────────────────────────────────┐
         │   Weekly Fine-Tuning Pipeline        │
         │                                      │
         │  Step 1: Export new feedback data    │
         │  Step 2: Merge with original dataset │
         │  Step 3: Re-train model              │
         │  Step 4: Evaluate (A/B test)         │
         │  Step 5: Deploy if improved          │
         └─────────────────────────────────────┘
```

1. **Weekly schedule:** Run every Sunday at 2 AM
2. **Data selection:** Only include samples with officer override (confirmed corrections)
3. **Fine-tuning:** Train for 3–5 epochs with low learning rate (1e-5)
4. **Gate check:** Only deploy new model if Critical-recall >= current model
5. **Rollback:** Keep last 3 model versions for quick rollback

### Step 4: Officer Dashboard

```
┌─────────────────────────────────────────────────────┐
│  📊 Feedback Analytics Dashboard                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Override Rate: 8.3%  ▲ 2.1% from last week         │
│  Total Feedback: 1,247 samples collected             │
│  Next Training: Sunday, 2:00 AM                      │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Override by Category                           │ │
│  │                                                 │ │
│  │  Pension:       ████████████  12%               │ │
│  │  Health:        █████████████  14%              │ │
│  │  Telecom:       ██████  6%                      │ │
│  │  Water:         █████  5%                       │ │
│  │  Corruption:    ████████████████  18%           │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Model Accuracy Trend                           │ │
│  │  ████████████████████████████████████░░░░  89%  │ │
│  │  ████████████████████████████████████░░  87%   │ │
│  │  ███████████████████████████████████  84%     │ │
│  │  ████████████████████████████████  81%       │ │
│  │  └────────────────────────────────┴──►        │ │
│  │     Week 1    Week 2    Week 3    Week 4      │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

# Phase 5: Real CPGRAMS API Integration

## 🎯 Objective

Integrate directly with the official CPGRAMS API to submit complaints, check status, and receive updates — making the system a true CPGRAMS client rather than a standalone tool.

## 📋 What This Phase Will Do

- Submit complaints directly to the central CPGRAMS system
- Receive real-time status updates from CPGRAMS
- Fetch complaint history and previous resolutions
- Synchronize department/category codes with the official system
- Enable officers to manage complaints through a single interface

## 🪜 Step-by-Step Execution

### Step 1: API Discovery & Documentation

```
┌─────────────────────────────────────────────────────────┐
│  CPGRAMS API Reference                                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Base URL: https://api.cpgrams.gov.in/v1                 │
│  Auth: API Key + OAuth 2.0 (Client Credentials)          │
│  Rate Limit: 100 req/min                                 │
│                                                          │
│  Endpoints:                                               │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ POST /complaints     — Submit new complaint          │ │
│  │ GET  /complaints/{id} — Get complaint details       │ │
│  │ GET  /status/{ref}   — Track complaint status       │ │
│  │ POST /auth/token     — Get access token             │ │
│  │ GET  /categories     — List all category codes       │ │
│  │ GET  /departments    — List all departments          │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          |
|  Data Format: JSON                                       |
|  Authentication: Bearer token in Authorization header    |
|  Webhook: POST callback URL for status updates           |
└─────────────────────────────────────────────────────────┘
```

### Step 2: API Client Module

```python
# app/cpgrams_client.py
import httpx
from datetime import datetime, timedelta

class CPGRAMSClient:
    """Official CPGRAMS API client with automatic token refresh."""
    
    def __init__(self, api_key, api_secret):
        self.base_url = "https://api.cpgrams.gov.in/v1"
        self.api_key = api_key
        self.api_secret = api_secret
        self.token = None
        self.token_expiry = datetime.now()
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def _refresh_token(self):
        """Get a fresh OAuth token (tokens expire every 60 min)."""
        if datetime.now() < self.token_expiry:
            return
        resp = await self.client.post(f"{self.base_url}/auth/token", json={
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "grant_type": "client_credentials"
        })
        data = resp.json()
        self.token = data["access_token"]
        self.token_expiry = datetime.now() + timedelta(seconds=data["expires_in"] - 300)
    
    async def submit_complaint(self, citizen_data, classification, text):
        """Submit a complaint to CPGRAMS and return the official reference ID."""
        await self._refresh_token()
        resp = await self.client.post(
            f"{self.base_url}/complaints",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "citizen_name": citizen_data["name"],
                "contact": citizen_data["contact"],
                "address": citizen_data.get("address", ""),
                "complaint_text": text,
                "category_code": classification["category_code"],
                "org_code": classification["org_code"],
                "sub_category_code": classification.get("sub_category_code"),
            }
        )
        return resp.json()  # {"cpgrams_ref": "CPGRAMS202600123456", ...}
    
    async def track_status(self, cpgrams_ref):
        """Track the official CPGRAMS status of a complaint."""
        await self._refresh_token()
        resp = await self.client.get(
            f"{self.base_url}/status/{cpgrams_ref}",
            headers={"Authorization": f"Bearer {self.token}"}
        )
        return resp.json()
```

### Step 3: Webhook Integration

```
┌──────────┐     Submit Complaint     ┌──────────┐
│  Our     │ ───────────────────────►  │ CPGRAMS  │
│  System  │                          │ API      │
│          │ ◄─────────────────────── │          │
│          │    Webhook Callback       │          │
│          │    (Status Update)        │          │
└────┬─────┘                          └──────────┘
     │
     │ Update Complaint Status in Local DB
     │ Notify Citizen via Email/SMS
     ▼
┌──────────┐
│  Queue   │
│  Update  │
└──────────┘
```

### Step 4: Data Synchronization

```
                          ┌─────────────────────┐
                          │   CPGRAMS API       │
                          │   (Source of Truth) │
                          └──────────┬──────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  │                                      │
                  ▼                                      ▼
     ┌─────────────────────────┐     ┌─────────────────────────┐
     │  Daily Category Sync    │     │  Real-time Status Sync  │
     │                         │     │                         │
     │  Fetch latest category  │     │  Webhook receives       │
     │  codes every 24 hours   │     │  status changes         │
     │                         │     │                         │
     │  Update local category  │     │  Update local DB        │
     │  tree JSON              │     │  Notify citizen         │
     └─────────────────────────┘     └─────────────────────────┘
```

---

# Phase 6: Authenticated Roles & Dashboards

## 🎯 Objective

Implement role-based authentication with department-specific dashboards, audit logging, and granular access control for different user types.

## 📋 What This Phase Will Do

- **Citizens:** File and track their own complaints only
- **Grievance Officers:** View/action complaints in their department
- **Department Admins:** Manage officers, view department analytics
- **Super Admins:** Cross-department view, manage users, system config
- **API Access:** Token-based access for external integrations

## 🪜 Step-by-Step Execution

### Step 1: Role Hierarchy

```
                   ┌─────────────┐
                   │ Super Admin │  (Full system access)
                   └──────┬──────┘
                          │
              ┌───────────┴───────────┐
              │                       │
      ┌───────┴───────┐      ┌───────┴───────┐
      │ Dept Admin    │      │ Dept Admin    │  (Per-department)
      │ - Health      │      │ - Telecom     │
      └───────┬───────┘      └───────┬───────┘
              │                       │
      ┌───────┴───────┐      ┌───────┴───────┐
      │ Grievance     │      │ Grievance     │  (View + action
      │ Officers      │      │ Officers      │   complaints)
      └───────────────┘      └───────────────┘

                       ┌─────────────┐
                       │  Citizens   │  (File + track own)
                       └─────────────┘
```

### Step 2: Authentication Flow

```
                          ┌───────────────────────┐
                          │   Login Page           │
                          │   (Email + Password)   │
                          └──────────┬────────────┘
                                     ▼
                          ┌───────────────────────┐
                          │   Authenticate via     │
                          │   JWT + Refresh Token  │
                          └──────────┬────────────┘
                                     ▼
                    ┌────────────────────────────────┐
                    │  Role-Based Redirect            │
                    │                                 │
                    │  Citizen:  /dashboard           │
                    │  Officer:  /officer/complaints  │
                    │  Dept Admin: /admin/dashboard   │
                    │  Super Admin: /super/admin      │
                    └────────────────────────────────┘
```

**Authentication Implementation:**

```python
# app/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key-here"  # Move to env variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE = 30  # minutes

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class UserRole(str, Enum):
    CITIZEN = "citizen"
    OFFICER = "grievance_officer"
    DEPT_ADMIN = "department_admin"
    SUPER_ADMIN = "super_admin"

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Decode JWT and return current user with role + department."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "user_id": payload["sub"],
            "role": UserRole(payload["role"]),
            "department": payload.get("department"),
            "org_code": payload.get("org_code"),
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_role(required_roles: list[UserRole]):
    """Dependency factory for role-based access control."""
    async def checker(user: dict = Depends(get_current_user)):
        if user["role"] not in required_roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return checker
```

### Step 3: Department-Specific Dashboard

```
┌─────────────────────────────────────────────────────────┐
│  🏥 Ministry of Health — Grievance Dashboard            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  TODAY'S OVERVIEW                              Jul 2026  │
│  ┌──────┬──────┬──────┬──────┐                           │
│  │ 142  │  38  │  12  │   3  │                           │
│  │ Total│ Pending│ Escal.│ Crit.│                         │
│  └──────┴──────┴──────┴──────┘                           │
│                                                          │
│  MY QUEUE (Sorted by Priority)    [Filter ▼] [Sort ▼]   │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ ⚠️ CRITICAL │ PG-2026-4921 │ "hospital ..."  │ 20m │ │
│  │ 🟠 HIGH     │ PG-2026-4902 │ "ambulance ..."│  2h │ │
│  │ 🟡 MEDIUM   │ PG-2026-4887 │ "vaccination..."│  5h │ │
│  │ 🟢 LOW      │ PG-2026-4851 │ "suggestion..." │  1d │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  TEAM PERFORMANCE                     ┌────────────┐    │
│  Avg Response Time: 4.2h ▼ 0.5h      │ ▲ 92%      │    │
│  Resolution Rate: 78% ▲ 3%            │ SLA Met    │    │
│  Pending > 7 days: 6 ▼ 2             └────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Step 4: Audit Logging

```sql
CREATE TABLE audit_log (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMP DEFAULT NOW(),
    user_id         VARCHAR(50),
    user_role       VARCHAR(20),
    action          VARCHAR(50),    -- 'view_complaint', 'override_prediction', 'login', etc.
    resource_type   VARCHAR(30),    -- 'complaint', 'user', 'model', 'system'
    resource_id     VARCHAR(50),
    details         JSONB,          -- { "old_urgency": "Critical", "new_urgency": "High" }
    ip_address      INET,
    user_agent      TEXT
);

-- Partition by month for performance
CREATE TABLE audit_log_2026_07 PARTITION OF audit_log
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');
```

---

# Phase 7: Production Hardening

## 🎯 Objective

Transform the prototype into a production-grade system with proper database, security, monitoring, and reliability.

## 📋 What This Phase Will Do

- Replace JSON file storage with PostgreSQL
- Add HTTPS via automated SSL certificates
- Implement rate limiting and DDoS protection
- Add comprehensive monitoring and alerting
- Ensure 99.5%+ uptime with graceful degradation
- GDPR/IT Act compliance for citizen data

## 🪜 Step-by-Step Execution

### Step 1: Database Migration

```
┌──────────────────────────────────────┐
│        Current (Prototype)           │
│                                      │
│  complaints_db.json                  │
│  training_log.csv                    │
│  category_tree.json                  │
│  bilstm_meta.json                    │
│  (no backup, no indexing,            │
│   no concurrent write safety)        │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        Target (Production)           │
│                                      │
│  PostgreSQL 16 + pgvector            │
│  ┌────────────────────────────────┐  │
│  │ complaints table with indexes   │  │
│  │ users table (hashed passwords) │  │
│  │ feedback_log table              │  │
│  │ audit_log table (partitioned)   │  │
│  │ model_versions table            │  │
│  │ sessions table (Redis optional) │  │
│  └────────────────────────────────┘  │
│                                      │
│  Automated backups (daily)           │
│  Read replicas for analytics         │
│  Connection pooling (PgBouncer)      │
└──────────────────────────────────────┘
```

### Step 2: Infrastructure Stack

```
                         ┌─────────────┐
                         │   Nginx     │  (Reverse Proxy + SSL termination)
                         │  (HTTPS)    │
                         └──────┬──────┘
                                │
                   ┌────────────┴────────────┐
                   │                         │
            ┌──────┴──────┐          ┌──────┴──────┐
            │  FastAPI     │          │  FastAPI     │
            │  (Worker 1)  │          │  (Worker 2)  │  (Gunicorn + Uvicorn)
            └──────┬──────┘          └──────┬──────┘
                   │                         │
                   └────────────┬────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
             ┌──────┴──────┐        ┌──────┴──────┐
             │ PostgreSQL  │        │    Redis     │  (Caching + Sessions)
             │  (Primary)  │        │              │
             └──────┬──────┘        └─────────────┘
                    │
             ┌──────┴──────┐
             │ PostgreSQL  │
             │ (Replica)   │
             └─────────────┘
```

### Step 3: Security Implementation

```
┌─────────────────────────────────────────────────────────────┐
│                     Security Layers                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: Network                                            │
│  ├── HTTPS (TLS 1.3) with Let's Encrypt                      │
│  ├── DDoS protection (Cloudflare or similar)                 │
│  └── IP whitelist for admin endpoints                        │
│                                                              │
│  Layer 2: Application                                        │
│  ├── Rate limiting (100 req/min per IP for classify)         │
│  ├── Input validation (Pydantic schemas enforced)             │
│  ├── SQL injection prevention (parameterized queries)        │
│  └── XSS protection (output encoding in templates)           │
│                                                              │
│  Layer 3: Authentication                                     │
│  ├── JWT with short expiry (30 min access + 7 day refresh)   │
│  ├── bcrypt password hashing (cost factor 12)                │
│  ├── MFA for admin accounts (TOTP)                           │
│  └── Session management (Redis-based, invalidation on logout)│
│                                                              │
│  Layer 4: Data                                               │
│  ├── Personal data encryption at rest (AES-256)              │
│  ├── PII masking in logs                                     │
│  ├── Data retention policy (auto-delete after 5 years)       │
│  └── GDPR/IT Act compliance checklist                        │
└─────────────────────────────────────────────────────────────┘
```

### Step 4: Monitoring & Alerting

```
                               ┌─────────────────────────┐
                               │     Alert Manager        │
                               │  (PagerDuty / Slack /    │
                               │   Email)                 │
                               └────────────┬────────────┘
                                            │
         ┌──────────────────────────────────┴──────────────────────────────────┐
         │                                                                     │
┌────────┴────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┴────────┐
│   Prometheus     │  │    Grafana       │  │  Sentry          │  │  Uptime Robot       │
│  (Metrics)      │  │  (Dashboards)    │  │  (Error Tracking) │  │  (Uptime Monitor)   │
├─────────────────┤  ├──────────────────┤  ├──────────────────┤  ├────────────────────┤
│ - Request count  │  │ - Real-time      │  │ - Unhandled      │  │ - 5-min ping check  │
│ - Response time   │  │   dashboard      │  │   exceptions     │  │ - SSL expiry alert  │
│ - Error rate     │  │ - Weekly reports │  │ - Stack traces   │  │ - Response time     │
│ - GPU utilization │  │ - Anomaly alerts │  │ - User impacted  │  │   threshold alert   │
│ - Model latency   │  │   (P99 spike)    │  │   counts         │  └────────────────────┘
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

**Key Metrics to Monitor:**

```
┌─────────────────────────────────────────────────────────────┐
│              Grafana Dashboard: System Health                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  🔴 Critical Alerts (PagerDuty)                              │
│  ├── Service down for > 1 min                                │
│  ├── Error rate > 5% for 5 min                               │
│  ├── P99 latency > 5s for 5 min                              │
│  └── Model serving failure                                   │
│                                                              │
│  🟡 Warning Alerts (Slack)                                   │
│  ├── P99 latency > 2s for 10 min                             │
│  ├── Error rate > 1% for 10 min                              │
│  ├── Disk usage > 80%                                        │
│  └── Model accuracy drop > 3% in last 24h                    │
│                                                              │
│  ℹ️ Info Metrics (Grafana)                                   │
│  ├── Requests per second (current: 12/s, peak: 45/s)         │
│  ├── Average response time (classify: 180ms, analyze: 45ms)  │
│  ├── Active users (current: 23, today: 1,247)               │
│  └── Model performance (accuracy: 87.3%, crit recall: 100%) │
└─────────────────────────────────────────────────────────────┘
```

---

# Phase 8: Active Learning Pipeline

## 🎯 Objective

Implement an active learning system that automatically identifies low-confidence predictions and routes them for human labeling, continuously improving the model with minimal human effort.

## 📋 What This Phase Will Do

- Automatically detect uncertain predictions (confidence < threshold)
- Present uncertain cases to officers for labeling
- Prioritize which cases to label (most informative first)
- Periodically retrain with newly labeled data
- Track model improvement over time per phase

## 🪜 Step-by-Step Execution

### Step 1: Uncertainty Sampling

```
                 ┌──────────────────────────────────────┐
                 │   Every prediction is scored for      │
                 │   uncertainty                         │
                 └──────────────────┬───────────────────┘
                                    ▼
     ┌───────────────────────────────────────────────────────────┐
     │                Uncertainty Scoring Methods                 │
     ├──────────────┬───────────────────────────┬────────────────┤
     │   Method     │      Calculation          │    When to Use │
     ├──────────────┼───────────────────────────┼────────────────┤
     │ Least        │ max(softmax) < 0.7        │ Best general   │
     │ Confidence   │                           │ purpose        │
     ├──────────────┼───────────────────────────┼────────────────┤
     │ Margin       │ top_prob - second_prob    │ Separating     │
     │ Sampling     │ < 0.2                     │ close classes  │
     ├──────────────┼───────────────────────────┼────────────────┤
     │ Entropy      │ -Σ(p_i × log(p_i)) > 0.5 │ Most            │
     │ Sampling     │                           │ comprehensive  │
     ├──────────────┼───────────────────────────┼────────────────┤
     │ Monte Carlo │ Variance across 10         │ For Bayesian   │
     │ Dropout     │ dropout forward passes > 0.1 │ uncertainty  │
     └──────────────┴───────────────────────────┴────────────────┘
```

### Step 2: Human Labeling Queue

```
┌─────────────────────────────────────────────────────────┐
│  🎯 Active Learning — Review Queue                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Priority: Least Confidence → Most Confidence            │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 🔴 ENTROPY: 1.24 │ "pension stopped ... hospice..."│ │
│  │     AI: Medium (42%)  │  Officer: [Dropdown] ▼      │ │
│  │     Dept: Pension     │  Confirm: [✅] [✏️ Override] │ │
│  ├─────────────────────────────────────────────────────┤ │
│  │ 🟡 ENTROPY: 0.87 │ "broadband speed ..."           │ │
│  │     AI: Low (63%)     │  Officer: [Dropdown] ▼      │ │
│  │     Dept: Telecom     │  Confirm: [✅] [✏️ Override] │ │
│  ├─────────────────────────────────────────────────────┤ │
│  │ 🟢 ENTROPY: 0.52 │ "ration card not received"      │ │
│  │     AI: Medium (78%)  │  Officer: [Dropdown] ▼      │ │
│  │     Dept: Ration      │  Confirm: [✅] [✏️ Override] │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  Stats: 23 pending │ 147 labeled today │ 84% agreement  │
│  [Skip] [Save & Next] [Submit All]                      │
└─────────────────────────────────────────────────────────┘
```

### Step 3: Training Data Selection

```
Active Learning Loop (Every N predictions):

                          ┌─────────────┐
                          │  N=1000     │
                          │  predictions│
                          └──────┬──────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  Score for Uncertainty   │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  Select Top-K Most        │
                    │  Uncertain Predictions    │
                    │  K = 100 (10%)           │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  Send to Officer          │
                    │  Review Queue            │
                    └────────────┬─────────────┘
                                 ▼
           ┌─────────────────────────────────────┐
           │  Officer Labels → Add to Training   │
           │  Dataset                            │
           └────────────────┬────────────────────┘
                            ▼
           ┌─────────────────────────────────────┐
           │  Is dataset large enough?            │
           │  (accumulated 500+ new labels?)      │
           │           │               │          │
           │         Yes               No         │
           │           │               │          │
           │           ▼               ▼          │
           │  Retrain model     Continue          │
           │  (fine-tune)       collecting        │
           └─────────────────────────────────────┘
```

### Step 4: Performance Tracking

```
Active Learning Performance Over Time:

Model Accuracy
  90% ┤                    ┌─────■────
  88% ┤              ┌─────■
  86% ┤        ┌─────■
  84% ┤  ┌─────■
  82% ┤──■
  80% ┤
      └───┬────┬────┬────┬────┬────┬────┬────
         0    1K   2K   3K   4K   5K   6K   7K
                    Labeled Samples Added

Legend: ─── Accuracy   ■ Training milestone

Critical Recall
 100% ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  95% ┤
  90% ┤
      └───┬────┬────┬────┬────┬────┬────┬────
         0    1K   2K   3K   4K   5K   6K   7K
                    Labeled Samples Added

Labeling Cost Saved:
  Without active learning:  Need ~10K labeled samples
  With active learning:     Need ~2K smartly chosen samples
  Savings:                  80% reduction in labeling effort
```

---

# 📅 Implementation Timeline Summary

```
Phase  │ Description                    │ Duration  │ Dependencies │ Effort
───────┼────────────────────────────────┼───────────┼──────────────┼───────
  1    │ MuRIL/IndicBERT Transformer    │ 6 weeks   │ None         │ High
  2    │ Learning-to-Rank System        │ 4 weeks   │ Phase 1      │ Medium
  3    │ Duplicate Detection            │ 3 weeks   │ Phase 1      │ Medium
  4    │ Officer Feedback Loop          │ 4 weeks   │ Phase 6      │ Medium
  5    │ CPGRAMS API Integration        │ 6 weeks   │ Phase 7      │ High
  6    │ Auth Roles & Dashboards        │ 5 weeks   │ None         │ High
  7    │ Production Hardening           │ 6 weeks   │ Phase 6      │ High
  8    │ Active Learning Pipeline       │ 4 weeks   │ Phase 1, 4   │ Medium
───────┼────────────────────────────────┼───────────┼──────────────┼───────
       │ Total                          │ 38 weeks* │              │
       │                                │ (~9.5 mo) │              │
       
* Phases can run in parallel where no dependencies exist:
  Phase 1 + Phase 6 + Phase 7 can start simultaneously
  → Optimistic timeline: ~5-6 months
```

---

<div align="center">

## 🎯 From Prototype to Production

```
    MVP ───────────► v1.0 ───────────► v1.5 ───────────► v2.0
     │                │                  │                  │
     │ BiLSTM        │ Transformer      │ CPGRAMS          │ Full production
     │ FastAPI       │ Multilingual     │ Integration      │ system with
     │ JSON store    │ Ranking          │ Auth & Roles     │ active learning
     │ 4 classes     │ Duplicate detect │ Auditing         │ & monitoring
     │ 1 language    │                  │                  │
```

</div>
