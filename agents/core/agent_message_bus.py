"""Simple async message bus for agent communication.

Agents register with the bus and receive an asyncio.Queue where messages
for that agent are delivered. The bus also supports send-and-wait semantics
where a sender can await a reply to a message ID with a timeout.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MessageBus:
    def __init__(self):
        # agent_name -> asyncio.Queue
        self.queues: Dict[str, asyncio.Queue] = {}
        # message_id -> asyncio.Future
        self._pending: Dict[str, asyncio.Future] = {}

    def register(self, agent_name: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self.queues[agent_name] = q
        logger.debug(f"Registered agent queue: {agent_name}")
        return q

    def unregister(self, agent_name: str) -> None:
        if agent_name in self.queues:
            del self.queues[agent_name]

    async def send(self, message: Dict[str, Any]) -> None:
        """Send a message to a single recipient specified in message['to']."""
        to = message.get("to")
        if not to:
            logger.warning("Message missing 'to' field: %s", message)
            return

        queue = self.queues.get(to)
        if not queue:
            logger.warning("No queue for recipient %s", to)
            return

        await queue.put(message)

    def send_nowait(self, message: Dict[str, Any]) -> None:
        to = message.get("to")
        if not to:
            logger.warning("Message missing 'to' field: %s", message)
            return
        queue = self.queues.get(to)
        if not queue:
            logger.warning("No queue for recipient %s", to)
            return
        queue.put_nowait(message)

    async def send_and_wait(self, message: Dict[str, Any], timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Send a message and wait for a reply addressed to message['from'] with reply_to set."""
        message_id = message.get("message_id")
        if not message_id:
            message_id = f"msg-{int(time.time()*1000)}"
            message["message_id"] = message_id

        future = asyncio.get_event_loop().create_future()
        self._pending[message_id] = future

        await self.send(message)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for reply to %s", message_id)
            return None
        finally:
            self._pending.pop(message_id, None)

    def notify_reply(self, reply_message: Dict[str, Any]) -> None:
        """Called by agents to fulfill a pending send_and_wait future.

        The reply must include a `reply_to` field equal to original message_id.
        """
        reply_to = reply_message.get("reply_to")
        if not reply_to:
            return
        fut = self._pending.get(reply_to)
        if fut and not fut.done():
            fut.set_result(reply_message)


def make_task_message(from_agent: str, to_agent: str, task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "message_id": f"{from_agent}-{task_id}-{int(time.time()*1000)}",
        "type": "task",
        "from": from_agent,
        "to": to_agent,
        "task_id": task_id,
        "payload": payload,
        "timestamp": time.time(),
    }


def make_result_message(from_agent: str, to_agent: str, task_id: str, payload: Dict[str, Any], reply_to: str) -> Dict[str, Any]:
    return {
        "message_id": f"{from_agent}-res-{task_id}-{int(time.time()*1000)}",
        "type": "result",
        "from": from_agent,
        "to": to_agent,
        "task_id": task_id,
        "payload": payload,
        "reply_to": reply_to,
        "timestamp": time.time(),
    }
