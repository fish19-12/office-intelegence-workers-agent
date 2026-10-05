"""Base agent class used by all specialists and the supervisor."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from agent_message_bus import MessageBus, make_result_message

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseAgent:
    def __init__(self, name: str, llm: Any, mcp: Any, bus: MessageBus, allowed_tools: Optional[list] = None):
        self.name = name
        self.llm = llm
        self.mcp = mcp
        self.bus = bus
        self.inbox = bus.register(name)
        self.allowed_tools = allowed_tools or []
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info(f"Agent {self.name} starting")
        asyncio.create_task(self._message_loop())

    async def stop(self) -> None:
        self._running = False
        self.bus.unregister(self.name)

    async def _message_loop(self) -> None:
        while self._running:
            try:
                msg = await self.inbox.get()
                await self.handle_message(msg)
            except Exception as exc:
                logger.exception("Agent %s loop error: %s", self.name, exc)

    async def handle_message(self, message: Dict[str, Any]) -> None:
        """Override in subclasses to handle incoming messages."""
        typ = message.get("type")
        if typ == "task":
            await self._handle_task(message)
        else:
            logger.debug("Unhandled message type %s for agent %s", typ, self.name)

    async def _handle_task(self, message: Dict[str, Any]) -> None:
        task_id = message.get("task_id")
        from_agent = message.get("from")
        try:
            payload = message.get("payload", {})
            result = await asyncio.to_thread(self.perform_task, payload)
            reply = make_result_message(self.name, from_agent, task_id, {"result": result}, reply_to=message.get("message_id"))
            # notify pending waiters
            self.bus.notify_reply(reply)
            self.bus.send_nowait(reply)
        except Exception as exc:
            reply = make_result_message(self.name, from_agent, task_id, {"error": str(exc)}, reply_to=message.get("message_id"))
            self.bus.notify_reply(reply)
            self.bus.send_nowait(reply)

    def perform_task(self, payload: Dict[str, Any]) -> Any:
        """Synchronous task performer; override with concrete logic in subclasses.

        Runs in a worker thread via asyncio.to_thread.
        """
        # Default: echo payload
        time.sleep(0.1)
        return {"echo": payload}
