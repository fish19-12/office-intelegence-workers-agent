from __future__ import annotations

from typing import Any, Dict

from agents.core.base_agent import BaseAgent


class SearchAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a search specialist using retrieval-augmented methods to find the most relevant evidence from documents. "
        "Prioritize accuracy, traceability, and context, and return grounded answers with citations or source-aware summaries whenever possible."
    )

    def perform_task(self, payload: Dict[str, Any]) -> Any:
        query = payload.get("query")
        top_k = int(payload.get("top_k", 5))
        if "document_search" in (self.allowed_tools or []):
            try:
                hits = self.mcp.call_tool("document_search", query=query, top_k=top_k)
                return {"hits": hits}
            except Exception:
                pass
        return {"query": query, "hits": []}
