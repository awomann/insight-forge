# IMPORTS

import json
from datetime import datetime


# REPORT GENERATION
# Packages insight, evidence, confidence, fairness into one structured dictionary

# Determines what is needed as inputs; packages insight, evidence, confidence, and fairness into a structured governance report
def generate_governance_report(
    question: str,
    insight: str,
    evidence: dict,
    confidence: dict,
    fairness: dict,
    assumptions: list,
    requested_by: str = "User"
) -> dict:
    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "requested_by": requested_by,
            "question": question
        },
        "insight": insight,
        "confidence": confidence,
        "evidence": evidence,
        "fairness": fairness,
        "assumptions": assumptions,
        "status": _determine_status(confidence, fairness)
    }

# Helper function that determines the report status
def _determine_status(confidence: dict, fairness: dict) -> str:
    score = confidence.get("overall_score", 1.0)
    flags = fairness.get("total_flags", 0)
    
    if score >= 0.75 and flags == 0:
        return "✅ Approved — high confidence, no fairness concerns"
    elif flags > 2 or score < 0.4:
        return "🔴 Escalate — significant concerns require human review"
    else:
        return "🟡 Review Further — moderate confidence or fairness flags present"


# FORMATTING
# Converts that dictionary into a readable markdown report for different audiences (executive, compliance, technical)

# Converts the governance report dictionary into readable markdown for different audiences
def format_report_markdown(report: dict) -> str:
    meta = report["metadata"]
    lines = [
        f"# Governance Report",
        f"**Generated:** {meta['generated_at']}",
        f"**Requested by:** {meta['requested_by']}",
        f"**Question:** {meta['question']}",
        "",
        f"## Status",
        report["status"],
        "",
        f"## Insight",
        report["insight"],
        "",
        f"## Confidence",
        f"**Level:** {report['confidence']['label']} ({report['confidence']['overall_score']})",
        "",
        f"## Fairness",
        report["fairness"]["summary"],
        "",
        f"## Assumptions",
    ]
    for a in report["assumptions"]:
        lines.append(f"- {a}")
    
    return "\n".join(lines)