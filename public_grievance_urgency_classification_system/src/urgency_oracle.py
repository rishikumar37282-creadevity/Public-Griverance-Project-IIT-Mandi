"""Weak-supervision urgency oracle (ported verbatim from the notebook).

Produces a 0-100 urgency score plus an explainable per-signal breakdown:
category severity, lexicon hits, temporal escalation, vulnerable groups,
structural (escalation depth). The label thresholds are:
Critical >= 70, High >= 45, Medium >= 25, else Low.
"""
import re

URGENCY_LEXICON = {
    "urgent": 8, "immediately": 8, "emergency": 10, "critical": 9, "no response": 7,
    "repeatedly": 6, "still not": 6, "unresolved": 6, "harass": 9, "threat": 9, "unsafe": 8,
    "danger": 9, "died": 10, "death": 10, "medical": 7, "hospital": 7, "elderly": 6,
    "disabled": 6, "child": 6, "senior citizen": 7, "pregnant": 7, "bribe": 8, "fraud": 8,
    "corruption": 7, "stolen": 6, "cheated": 6, "stopped": 5, "denied": 5, "pension": 3,
}
VULNERABLE = {"elderly", "senior citizen", "disabled", "child", "children", "pregnant", "patient", "widow"}
SEVERE = {"fraud", "corruption", "harass", "safety", "no service", "stopped", "denied", "illegal", "bribe"}
LOWSEV = {"suggestion", "revision", "enquiry", "query", "information"}

LABELS = ["Low", "Medium", "High", "Critical"]


def urgency_score(text: str, category_desc: str = "", escalation_depth: int = 0):
    """Return (total 0-100, label, parts dict, matched keywords)."""
    t = (text or "").lower()
    parts = {}
    d = (category_desc or "").lower()
    parts["category"] = 22 if any(k in d for k in SEVERE) else (2 if any(k in d for k in LOWSEV) else 10)

    hits = [k for k in URGENCY_LEXICON if k in t]
    parts["lexicon"] = min(sum(URGENCY_LEXICON[k] for k in hits), 30)

    temp = 0
    for n, unit in re.findall(r"(\d+)\s*(day|week|month|year)", t):
        n = int(n)
        if unit == "day" and n >= 7:
            temp += 4
        if unit == "week" and n >= 2:
            temp += 5
        if unit in ("month", "year"):
            temp += 7
    parts["temporal"] = min(temp, 20)

    vulnerable_hits = [v for v in VULNERABLE if v in t]
    parts["vulnerable"] = 10 if vulnerable_hits else 0
    parts["structural"] = min(escalation_depth * 3, 15)

    total = min(sum(parts.values()), 100)
    label = "Critical" if total >= 70 else "High" if total >= 45 else "Medium" if total >= 25 else "Low"
    return total, label, parts, {"lexicon": hits, "vulnerable": vulnerable_hits}
