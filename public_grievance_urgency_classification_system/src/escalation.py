"""Escalation recommendation + actionable next steps (decision-support output).

Given the predicted urgency and department, produce the structured guidance
the project brief asks for: routing level, expected first-response and
resolution SLAs, concrete next steps for the citizen, and the officer-side
escalation path. SLA values are illustrative policy defaults, not statutory.
"""

ESCALATION_MATRIX = {
    "Critical": {
        "priority_rank": 1,
        "routing_level": "Senior Officer — immediate desk",
        "first_response": "Within 24 hours",
        "resolution_target": "3 working days",
        "escalation_path": [
            "Auto-flagged to the department's Nodal Grievance Officer immediately",
            "If no action in 24 h: escalate to Director/Joint Secretary level",
            "If no action in 3 days: escalate to DARPG central monitoring (CPGRAMS)",
        ],
        "citizen_steps": [
            "Keep your reference ID safe — quote it in every follow-up.",
            "If there is immediate danger to life or safety, also call the relevant emergency helpline (112).",
            "Attach any evidence (photos, receipts, medical papers) as soon as possible.",
            "You should receive an officer contact within 24 hours; if not, use the Track page to escalate.",
        ],
        "note": "Flagged as an emergency. A human officer must review this within one working day.",
    },
    "High": {
        "priority_rank": 2,
        "routing_level": "Section Officer — priority queue",
        "first_response": "Within 3 working days",
        "resolution_target": "15 working days",
        "escalation_path": [
            "Routed to the concerned section's priority queue",
            "If no acknowledgement in 3 days: escalate to Nodal Grievance Officer",
            "If unresolved in 15 days: eligible for reminder + appeal",
        ],
        "citizen_steps": [
            "Note your reference ID and expected response date.",
            "Gather supporting documents (previous complaint numbers, bills, ID proofs).",
            "Check status after 3 working days on the Track page.",
            "If the response is unsatisfactory, file an appeal citing this reference ID.",
        ],
        "note": "High priority. Serious service failure or safety concern detected.",
    },
    "Medium": {
        "priority_rank": 3,
        "routing_level": "Dealing Assistant — standard queue",
        "first_response": "Within 7 working days",
        "resolution_target": "30 working days",
        "escalation_path": [
            "Routed to the department's standard grievance queue",
            "Reminder auto-generated at 15 days without action",
            "Appeal available after 30 days",
        ],
        "citizen_steps": [
            "Save your reference ID.",
            "Track progress weekly on the Track page.",
            "Add any new information to the same reference — do not file duplicates.",
            "After 30 days without resolution you may escalate to the appellate authority.",
        ],
        "note": "Standard processing. Persistent issues are monitored for escalation.",
    },
    "Low": {
        "priority_rank": 4,
        "routing_level": "General queue / feedback desk",
        "first_response": "Within 15 working days",
        "resolution_target": "45 working days",
        "escalation_path": [
            "Logged in the general queue for the concerned department",
            "Suggestions and information requests are compiled monthly for review",
        ],
        "citizen_steps": [
            "Save your reference ID for future correspondence.",
            "For information requests, also consider filing an RTI application for a statutory timeline.",
            "Check the Track page after two weeks.",
        ],
        "note": "Routine request or feedback. Thank you for helping improve public services.",
    },
}


def recommend(urgency: str, department: str, org_code: str | None = None):
    plan = ESCALATION_MATRIX.get(urgency, ESCALATION_MATRIX["Medium"]).copy()
    plan["urgency"] = urgency
    plan["department"] = department
    plan["org_code"] = org_code
    plan["routed_to"] = f"{department} ({org_code})" if org_code else department
    return plan
