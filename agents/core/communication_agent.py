from __future__ import annotations

from typing import Any, Dict

from agents.core.base_agent import BaseAgent


class CommunicationAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a communication specialist focused on clear, persuasive, and professional messaging. "
        "Write emails and notifications that are concise, action-oriented, and aligned with the user's business goal. "
        "Tailor tone to the audience, clarify any missing context, and ensure the message is easy to act on."
    )

    def perform_task(self, payload: Dict[str, Any]) -> Any:
        channel = payload.get("channel", "email")
        message = payload.get("message", "")
        recipients = payload.get("recipients", [])
        if channel == "email" and "email_send" in (self.allowed_tools or []):
            try:
                res = self.mcp.call_tool("email_send", recipients=recipients, message=message)
                return {"sent": True, "tool_result": res}
            except Exception:
                raise
        return {"channel": channel, "recipients": recipients, "message_preview": message[:120]}
