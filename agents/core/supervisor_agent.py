"""Supervisor agent that orchestrates tasks across specialist agents."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent_message_bus import MessageBus, make_task_message
from base_agent import BaseAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SupervisorAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a supervisory orchestration agent responsible for decomposing complex goals into executable tasks and assigning them to the right specialists. "
        "Maintain clarity of objectives, minimize duplicate effort, and prefer the simplest reliable plan that achieves the user's intent."
    )

    def __init__(self, name: str, llm: Any, mcp: Any, bus: MessageBus, specialists: Dict[str, Dict[str, Any]]):
        super().__init__(name, llm, mcp, bus, allowed_tools=[])
        # specialists map name -> {allowed_tools: [...]} used for simple validation
        self.specialists = specialists

    async def dispatch_user_goal(self, user_goal: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Decompose goal via LLM, assign tasks to specialists, wait for results with timeouts."""
        context = context or {}
        # Ask LLM to decompose into JSON tasks: [{agent: 'data', task_id:'t1', payload: {...}}, ...]
        prompt = (
            f"Supervisor: break the user goal into subtasks and assign to specialists.\nGoal: {user_goal}\n\n"
            f"Specialists: {list(self.specialists.keys())}\n\n"
            "Return ONLY a JSON array of tasks: [{\"agent\": \"DataAgent\", \"task_id\": \"t1\", \"payload\": {...}}, ...]"
        )

        raw = None
        try:
            raw = self.llm.generate(prompt)
            tasks = json.loads(raw)
        except Exception:
            # Fallback heuristic decomposition for common goals
            tasks = self._fallback_decompose(user_goal)

        # Send tasks in parallel and await replies
        results = {}
        coros = []
        for task in tasks:
            agent_name = task.get("agent")
            task_id = task.get("task_id")
            payload = task.get("payload", {})
            if agent_name not in self.specialists:
                logger.warning("Unknown specialist %s; skipping task %s", agent_name, task_id)
                results[task_id] = {"error": "unknown specialist"}
                continue

            msg = make_task_message(self.name, agent_name, task_id, payload)
            coro = self._send_task_and_wait(msg, timeout=30.0)
            coros.append((task_id, agent_name, coro))

        # execute coroutines concurrently and collect
        for task_id, agent_name, coro in coros:
            try:
                reply = await asyncio.wait_for(coro, timeout=30.0)
                results[task_id] = reply
            except asyncio.TimeoutError:
                logger.warning("Task %s to %s timed out; marking as timed_out", task_id, agent_name)
                results[task_id] = {"error": "timeout"}

        # Combine results (simple merge)
        combined = {k: v for k, v in results.items()}
        return {"tasks": tasks, "results": combined}

    async def _send_task_and_wait(self, message: Dict[str, Any], timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        # Use MessageBus send_and_wait so pending futures are resolved by agents
        return await self.bus.send_and_wait(message, timeout=timeout)

    def _fallback_decompose(self, goal: str) -> List[Dict[str, Any]]:
        # Very small heuristic: map keywords to agents
        tasks = []
        if "report" in goal.lower():
            tasks.append({"agent": "ReportAgent", "task_id": "r1", "payload": {"report_name": "auto.docx", "sections": []}})
        if "email" in goal.lower() or "notify" in goal.lower():
            tasks.append({"agent": "CommunicationAgent", "task_id": "c1", "payload": {"channel": "email", "message": goal, "recipients": []}})
        if "loan" in goal.lower() or "risk" in goal.lower():
            tasks.append({"agent": "RiskAgent", "task_id": "risk1", "payload": {"application": {"amount": 60000}}})
        if not tasks:
            tasks.append({"agent": "SearchAgent", "task_id": "s1", "payload": {"query": goal}})
        return tasks
