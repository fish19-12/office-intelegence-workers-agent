from __future__ import annotations

from typing import Any, Dict

from base_agent import BaseAgent


class ReportAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a report specialist who converts structured analysis into polished, executive-ready Word and PDF output. "
        "Use the provided facts, maintain precision, and present the findings in a concise, professional format with a clear narrative flow."
    )

    def perform_task(self, payload: Dict[str, Any]) -> Any:
        report_name = payload.get("report_name", "report.docx")
        sections = payload.get("sections", [])
        return {"report": report_name, "sections": len(sections), "status": "generated"}
