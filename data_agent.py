from __future__ import annotations

from typing import Any, Dict

from base_agent import BaseAgent


class DataAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a data specialist who transforms raw spreadsheet content into reliable analytical outputs. "
        "Clean inconsistent data, compute accurate aggregates, explain assumptions, and surface quality issues before making recommendations."
    )

    def perform_task(self, payload: Dict[str, Any]) -> Any:
        sheet_url = payload.get("sheet_url")
        operation = payload.get("operation", "summarize")

        if sheet_url and "google_sheets_read" in (self.allowed_tools or []):
            try:
                data = self.mcp.call_tool("google_sheets_read", sheet_url=sheet_url)
            except Exception:
                raise
        else:
            data = payload.get("data", [])

        if operation == "summarize":
            return {"rows": len(data), "preview": data[:3]}
        return {"data": data}
