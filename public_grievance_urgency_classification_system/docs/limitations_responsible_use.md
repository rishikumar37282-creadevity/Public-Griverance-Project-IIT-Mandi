# Limitations & Responsible Use

## Limitations

1. **Data scope.** Trained on ~4,840 English complaints covering 15 departments. Complaints in
   Hindi or regional languages, code-mixed text, or about unlisted departments will be
   misrouted or classified with low confidence. Out-of-vocabulary words fall back to `<UNK>`.
2. **Weak-supervision labels.** Urgency labels come from a rule-based oracle, not human triage
   experts. The neural model can only be as good as those rules; real deployments need
   officer-verified labels.
3. **Short-text bias.** Dataset complaints average ~13 tokens; very long, narrative complaints
   are truncated at `max_len` tokens.
4. **Illustrative SLAs.** First-response/resolution timelines and escalation paths in the action
   plan are policy defaults for demonstration, not statutory commitments.
5. **Attention ≠ full explanation.** Attention weights show where the model looked, which is
   useful but not a formal causal explanation of the decision.
6. **Prototype persistence/security.** Complaints are stored in a local JSON file; there is no
   authentication on the admin app. Both are deliberate simplifications for an academic
   prototype and must be replaced (database, RBAC, audit logging, HTTPS) before any real use.

## Responsible use

- **Human in the loop.** Predictions are advisory. Every complaint is queued for officer
  review; the urgency level orders the queue, it never closes or rejects a complaint.
- **Emergency guardrail.** The UI repeatedly instructs users to call **112** for
  life-threatening situations — an AI triage queue is not an emergency channel.
- **Asymmetric-error design.** The loss is deliberately biased toward catching Critical cases
  (recall 1.00 on test at the cost of some false Critical alarms) because a missed emergency is
  far worse than an over-prioritised one.
- **Privacy.** Personal details are optional and anonymous filing is allowed; the public track
  endpoint never returns contact details; nothing leaves the local server.
- **Transparency.** Citizens see the score breakdown and the attention highlights behind every
  decision; admins can audit every training run in `logs/training_log.csv`.
- **No punitive automation.** Outputs must never be used to penalise complainants or to make
  legally binding decisions without human judgement.
