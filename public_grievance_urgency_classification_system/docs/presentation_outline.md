# Presentation Outline (8–10 slides)

1. **Title & team** — Jan Samadhan: Public Grievance Urgency Classification System.
   AIML track · BiLSTM + Attention · two-platform web app. Team: Rishi Kumar (+ members).
2. **Problem & real-world impact** — one queue for everything means emergencies wait behind
   suggestions. Faster triage = faster response for pensioners, patients, safety complaints.
3. **Dataset** — Government of India Grievance Report (Kaggle): 4,840 complaints, 15
   departments with CPGRAMS codes; official 19,853-node category tree from
   CategoryCode_Mapping.xlsx; Critical class only 2.4% → imbalance is the core challenge.
4. **System workflow** — text → tokenise/pad → BiLSTM + attention → two heads (category,
   urgency) → oracle score 0–100 → escalation engine → citizen-facing action plan.
   (Show the pipeline diagram from the "How It Works" tab.)
5. **AI innovation** — multi-task learning; class-weighted loss with critical_boost ×3;
   checkpoint & early-stopping on Critical-recall (+ macro-F1 tie-break); attention-based
   explainability; weak-supervision oracle with explainable parts.
6. **Demo — citizen portal** — live triage preview while typing, voice input, urgency gauge,
   attention highlights, CPGRAMS sub-category suggestion, action plan + SLAs, tracking.
7. **Demo — Model Observatory** — Architecture Explorer (real forward pass, layer by layer),
   Training Studio (all hyperparameters live, per-epoch curves + confusion matrix,
   training_log.csv), Experiments comparison (show the no-critical-boost ablation).
8. **Results** — test set (726): Critical recall 1.000, macro-F1 0.886, urgency accuracy
   87.3%, category accuracy 100%. Ablation: without the boost, 6% of emergencies are missed.
9. **Limitations & responsible use** — English-only, weak labels, advisory-only with human
   review, emergency guardrail (call 112), privacy, prototype security caveats.
10. **Future improvements & conclusion** — multilingual transformer (MuRIL), officer feedback
    loop, duplicate detection, real CPGRAMS integration.

**Demo video plan (5–8 min):** problem (1 min) → dataset (30 s) → citizen flow demo (2 min)
→ admin observatory demo (2 min: explainer, live training run, experiment comparison)
→ results + limitations (1 min).
