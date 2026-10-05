"""
Reactive Planning Module

Implements a DAG-based planner for microfinance workflows.
Supports dependency graphs, parallel execution, conditional branching,
dynamic replanning after failures, and simple plan visualization.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from context_retriever import ContextRetriever, RetrievedChunk
from memory_manager import MemoryManager
from mcp_manager import MCPError, MCPManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PlanStep:
    """Represents a single step in a DAG execution plan."""

    step_num: int
    name: str
    tool_name: Optional[str]
    args: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)
    condition: Optional[str] = None
    timeout_seconds: int = 60
    retry_count: int = 0
    rationale: str = ""
    status: str = "pending"
    result: Optional[Any] = None
    duration_ms: Optional[int] = None
    error_msg: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_num": self.step_num,
            "name": self.name,
            "tool_to_use": self.tool_name,
            "args": self.args,
            "depends_on": self.depends_on,
            "condition": self.condition,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "rationale": self.rationale,
            "status": self.status,
            "result": self.result,
            "duration_ms": self.duration_ms,
            "error_msg": self.error_msg,
        }


@dataclass
class PlanResult:
    """Result of a complete DAG execution."""

    answer: str
    steps: List[PlanStep]
    partial: bool = False
    goal_achieved: Optional[bool] = None
    total_duration_ms: Optional[int] = None
    diagram: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": [step.to_dict() for step in self.steps],
            "partial": self.partial,
            "goal_achieved": self.goal_achieved,
            "total_duration_ms": self.total_duration_ms,
            "diagram": self.diagram,
        }


# ============================================================================
# REACTIVE PLANNER
# ============================================================================

class ReactivePlanner:
    """DAG-based planner with parallel execution and dynamic replanning."""

    def __init__(
        self,
        llm: Any,
        mcp: MCPManager,
        memory: MemoryManager,
        retriever: ContextRetriever,
    ):
        self.llm = llm
        self.mcp = mcp
        self.memory = memory
        self.retriever = retriever
        self.MAX_STEPS = 10
        self.MAX_STEPS_HARD_CAP = 20

    def run(
        self,
        goal: str,
        context_chunks: List[RetrievedChunk],
        max_steps: Optional[int] = None,
    ) -> PlanResult:
        max_steps = min(max_steps or self.MAX_STEPS, self.MAX_STEPS_HARD_CAP)
        start_time = time.time()

        plan_steps = self._plan_phase(goal, context_chunks)
        diagram = self._render_plan_diagram(plan_steps)
        completed = self._execute_plan(goal, context_chunks, plan_steps, max_steps)

        plan_result = self._verify_phase(goal, context_chunks, completed)
        plan_result.diagram = diagram
        plan_result.total_duration_ms = int((time.time() - start_time) * 1000)
        plan_result.partial = len(completed) < len(plan_steps) or plan_result.partial

        logger.info(f"Planning complete. Duration: {plan_result.total_duration_ms}ms")
        return plan_result

    def _plan_phase(self, goal: str, context_chunks: List[RetrievedChunk]) -> List[PlanStep]:
        context = self.retriever.format_for_prompt(context_chunks)
        memory_ctx = self.memory.get_prompt_context()
        tool_schemas = self._format_tool_schemas()

        prompt = (
            f"You are a microfinance agent. Goal: {goal}\n\n"
            f"Available context:\n{context}\n\n"
            f"Previous memory:\n{memory_ctx}\n\n"
            f"Available tools:\n{tool_schemas}\n\n"
            "Produce a DAG execution plan as a JSON array. Each step must include:\n"
            "  step, name, tool_to_use, args, depends_on, condition (optional), timeout_seconds, retry_count, rationale\n"
            "If the answer can be produced directly, return a single step with tool_to_use=null.\n"
            "Do not include any text outside the JSON array.\n"
            "Example:\n"
            "[\n"
            "  {\n"
            "    \"step\": 1,\n"
            "    \"name\": \"Load data\",\n"
            "    \"tool_to_use\": \"google_sheets_read\",\n"
            "    \"args\": {\"sheet_url\": \"...\"},\n"
            "    \"depends_on\": [],\n"
            "    \"condition\": null,\n"
            "    \"timeout_seconds\": 120,\n"
            "    \"retry_count\": 1,\n"
            "    \"rationale\": \"Load data before generating the report.\"\n"
            "  }\n"
            "]"
        )

        raw = self.llm.generate(prompt)
        plan_data = self._parse_json(raw)
        if plan_data is None:
            raw = self.llm.generate("The previous reply was not valid JSON. Reply ONLY with the JSON array plan." + prompt)
            plan_data = self._parse_json(raw)

        if not isinstance(plan_data, list):
            logger.warning("Plan parsing failed; falling back to direct answer")
            return [self._fallback_direct_step()]

        plan_steps = self._parse_plan_json(plan_data)
        if not self._validate_plan_graph(plan_steps):
            logger.warning("Generated plan is not a valid DAG; falling back to direct answer")
            return [self._fallback_direct_step()]

        return plan_steps

    def _execute_plan(
        self,
        goal: str,
        context_chunks: List[RetrievedChunk],
        plan_steps: List[PlanStep],
        max_steps: int,
    ) -> Dict[int, PlanStep]:
        pending = {step.step_num: step for step in plan_steps}
        completed: Dict[int, PlanStep] = {}
        replanned = False

        while pending and len(completed) < max_steps:
            ready = [step for step in pending.values() if self._is_step_ready(step, completed)]
            if not ready:
                logger.info("No ready steps remain.")
                break

            if len(completed) + len(ready) > max_steps:
                ready = ready[: max_steps - len(completed)]

            executed = self._run_parallel_steps(ready, goal, context_chunks)
            for step in executed:
                pending.pop(step.step_num, None)
                completed[step.step_num] = step

            failed = [step for step in executed if step.status == "error"]
            if failed and not replanned:
                logger.info("Failure detected; attempting dynamic replanning")
                new_plan = self._replan_after_failure(goal, context_chunks, completed, list(pending.values()), failed)
                if new_plan:
                    pending = {step.step_num: step for step in new_plan if step.step_num not in completed}
                    replanned = True
                    continue

        return completed

    def _run_parallel_steps(self, ready_steps: List[PlanStep], goal: str, context_chunks: List[RetrievedChunk]) -> List[PlanStep]:
        async def run_all() -> List[PlanStep]:
            tasks = [asyncio.create_task(self._execute_step(step, goal, context_chunks)) for step in ready_steps]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            completed_steps: List[PlanStep] = []
            for step, result in zip(ready_steps, results):
                if isinstance(result, Exception) and step.status == "running":
                    step.status = "error"
                    step.error_msg = str(result)
                completed_steps.append(step)
            return completed_steps

        return asyncio.run(run_all())

    async def _execute_step(self, step: PlanStep, goal: str, context_chunks: List[RetrievedChunk]) -> PlanStep:
        step.status = "running"

        if not self._evaluate_step_condition(step, context_chunks):
            step.status = "skipped"
            step.error_msg = f"Skipped because condition evaluated to false: {step.condition}"
            return step

        for attempt in range(1, max(step.retry_count, 0) + 2):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._run_step, step, goal, context_chunks),
                    timeout=step.timeout_seconds,
                )
            except asyncio.TimeoutError:
                step.status = "error"
                step.error_msg = f"Timeout after {step.timeout_seconds}s"
                step.duration_ms = int((time.time() - (step.duration_ms or time.time())) * 1000)
                logger.error(f"Step {step.step_num} timed out")
                break
            except Exception as exc:
                step.status = "error"
                step.error_msg = str(exc)
                logger.error(f"Step {step.step_num} attempt {attempt} failed: {exc}")
                if attempt <= step.retry_count:
                    logger.info(f"Retrying step {step.step_num} immediately")
                    continue
                break

        return step

    def _run_step(self, step: PlanStep, goal: str, context_chunks: List[RetrievedChunk]) -> PlanStep:
        start_time = time.time()

        if step.tool_name is None:
            prompt = (
                f"Goal: {goal}\n\n"
                f"Context:\n{self.retriever.format_for_prompt(context_chunks)}\n\n"
                f"Memory:\n{self.memory.get_prompt_context()}\n\n"
                "Answer the goal directly based on the available context and memory."
            )
            answer = self.llm.generate(prompt)
            step.result = {"answer": answer}
            step.status = "success"
            step.duration_ms = int((time.time() - start_time) * 1000)
            return step

        entity_ids = self._extract_entity_ids(step.args)
        for entity_id in entity_ids:
            if self.memory.has_action_been_taken(step.tool_name, entity_id):
                step.status = "skipped"
                step.error_msg = f"Skipped: {step.tool_name} already executed for {entity_id}"
                step.duration_ms = int((time.time() - start_time) * 1000)
                return step

        try:
            result = self.mcp.call_tool(step.tool_name, **step.args)
            step.result = result
            step.status = "success"
            step.duration_ms = int((time.time() - start_time) * 1000)
            self.memory.record(
                "tool_executed",
                f"{step.tool_name} executed for step {step.step_num}",
                detail={"args": step.args, "result": result},
                entity_ids=entity_ids,
            )
            return step
        except MCPError as exc:
            step.status = "error"
            step.error_msg = str(exc)
            step.duration_ms = int((time.time() - start_time) * 1000)
            self.memory.record(
                "tool_error",
                f"{step.tool_name} failed for step {step.step_num}",
                detail={"args": step.args, "error": str(exc)},
                entity_ids=entity_ids,
            )
            return step

    def _replan_after_failure(
        self,
        goal: str,
        context_chunks: List[RetrievedChunk],
        completed: Dict[int, PlanStep],
        remaining: List[PlanStep],
        failed_steps: List[PlanStep],
    ) -> Optional[List[PlanStep]]:
        prompt = (
            f"A plan was running for goal: {goal}\n\n"
            f"Completed steps:\n{self._format_step_summary(sorted(completed.values(), key=lambda s: s.step_num))}\n\n"
            f"Remaining steps before replanning:\n{self._format_step_summary(sorted(remaining, key=lambda s: s.step_num))}\n\n"
            f"Failed steps:\n{self._format_step_summary(sorted(failed_steps, key=lambda s: s.step_num))}\n\n"
            "Generate a new DAG plan for the remaining work starting after the completed steps. "
            "Return ONLY a JSON array of steps. If no further actions are needed, return an empty array."
        )

        raw = self.llm.generate(prompt)
        plan_data = self._parse_json(raw)
        if not isinstance(plan_data, list):
            logger.warning("Replan output was not valid JSON")
            return None

        start_step = max(completed.keys(), default=0) + 1
        new_plan = self._parse_plan_json(plan_data, start_step=start_step)
        if not self._validate_plan_graph(new_plan):
            logger.warning("Replan produced invalid DAG")
            return None

        return new_plan

    def _verify_phase(
        self,
        goal: str,
        context_chunks: List[RetrievedChunk],
        completed: Dict[int, PlanStep],
    ) -> PlanResult:
        prompt = (
            f"Goal: {goal}\n\n"
            f"Executed steps:\n{self._format_step_summary(sorted(completed.values(), key=lambda s: s.step_num))}\n\n"
            "Was the goal fully achieved? Reply with JSON:\n"
            '{"achieved": true/false, "answer": "final answer", "missing": "what remains"}'
        )
        raw = self.llm.generate(prompt)
        verification = self._parse_json(raw)
        if not isinstance(verification, dict):
            return PlanResult(
                answer=raw,
                steps=list(completed.values()),
                goal_achieved=any(step.status == "success" for step in completed.values()),
            )

        return PlanResult(
            answer=verification.get("answer", "Task completed."),
            steps=list(completed.values()),
            goal_achieved=verification.get("achieved", True),
            partial=False,
        )

    def _is_step_ready(self, step: PlanStep, completed: Dict[int, PlanStep]) -> bool:
        if step.status != "pending":
            return False

        for dependency in step.depends_on:
            dep = completed.get(dependency)
            if dep is None:
                return False
            if dep.status != "success":
                step.status = "skipped"
                step.error_msg = f"Skipped: dependency {dependency} did not succeed"
                return False

        return True

    def _evaluate_step_condition(self, step: PlanStep, context_chunks: List[RetrievedChunk]) -> bool:
        if not step.condition:
            return True

        variables = self._build_condition_context(step)
        try:
            return bool(self._safe_eval(step.condition, variables))
        except Exception as exc:
            logger.warning(f"Condition failed for step {step.step_num}: {exc}")
            return False

    def _build_condition_context(self, step: PlanStep) -> Dict[str, Any]:
        variables: Dict[str, Any] = {}
        for key, value in step.args.items():
            if isinstance(value, (int, float, bool, str)):
                variables[key] = value
        return variables

    def _safe_eval(self, expression: str, variables: Dict[str, Any]) -> Any:
        node = ast.parse(expression, mode="eval")
        allowed = (
            ast.Expression,
            ast.BoolOp,
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.And,
            ast.Or,
            ast.Not,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.FloorDiv,
            ast.USub,
            ast.UAdd,
        )
        for item in ast.walk(node):
            if not isinstance(item, allowed):
                raise ValueError(f"Unsupported expression element: {type(item).__name__}")
        return eval(compile(node, filename="<condition>", mode="eval"), {"__builtins__": {}}, variables)

    def _render_plan_diagram(self, plan_steps: List[PlanStep]) -> str:
        if not plan_steps:
            return "[No plan generated]"

        levels = self._assign_plan_levels(plan_steps)
        lines = ["Plan diagram:"]
        for level in sorted(levels):
            entries = []
            for step in sorted(levels[level], key=lambda s: s.step_num):
                cond = f" if {step.condition}" if step.condition else ""
                entries.append(f"[{step.step_num}] {step.name} -> deps={step.depends_on}{cond}")
            lines.append(f"  Parallel group {level}: " + ", ".join(entries))
        return "\n".join(lines)

    def _assign_plan_levels(self, plan_steps: List[PlanStep]) -> Dict[int, List[PlanStep]]:
        step_map = {step.step_num: step for step in plan_steps}
        levels: Dict[int, List[PlanStep]] = {}

        def compute_level(step: PlanStep, visited: Optional[set] = None) -> int:
            if visited is None:
                visited = set()
            if step.step_num in visited:
                return 1
            visited.add(step.step_num)
            if not step.depends_on:
                return 1
            return 1 + max(
                compute_level(step_map[dep], visited)
                for dep in step.depends_on
                if dep in step_map
            )

        for step in plan_steps:
            level = compute_level(step)
            levels.setdefault(level, []).append(step)
        return levels

    def _validate_plan_graph(self, plan_steps: List[PlanStep]) -> bool:
        step_map = {step.step_num: step for step in plan_steps}
        visited: Dict[int, str] = {}

        def visit(step_num: int) -> bool:
            state = visited.get(step_num)
            if state == "visiting":
                return False
            if state == "visited":
                return True
            visited[step_num] = "visiting"
            step = step_map.get(step_num)
            if step is None:
                return False
            for dep in step.depends_on:
                if dep not in step_map:
                    return False
                if not visit(dep):
                    return False
            visited[step_num] = "visited"
            return True

        return all(visit(step.step_num) for step in plan_steps)

    def _format_tool_schemas(self) -> str:
        tools_list = self.mcp.registry.list()
        lines: List[str] = []
        for tool_name, tool_info in tools_list.items():
            desc = tool_info.get("description", "")
            schema = tool_info.get("schema", {})
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            if properties:
                arg_names = [f"{name} (required)" if name in required else name for name in properties.keys()]
                args_str = ", ".join(arg_names)
            else:
                args_str = "no args"
            lines.append(f"- {tool_name}: {desc} | args: {args_str}")
        return "\n".join(lines)

    def _parse_json(self, raw_text: str) -> Optional[Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            cleaned = raw_text.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    return None
            return None

    def _parse_plan_json(self, plan_data: List[Dict[str, Any]], start_step: int = 1) -> List[PlanStep]:
        steps: List[PlanStep] = []
        for index, item in enumerate(plan_data):
            step_num = item.get("step", start_step + index)
            tool_name = item.get("tool_to_use") if item.get("tool_to_use") is not None else item.get("tool")
            steps.append(
                PlanStep(
                    step_num=step_num,
                    name=item.get("name", f"Step {step_num}"),
                    tool_name=tool_name,
                    args=item.get("args", {}),
                    depends_on=item.get("depends_on", []),
                    condition=item.get("condition"),
                    timeout_seconds=item.get("timeout_seconds", 60),
                    retry_count=item.get("retry_count", 0),
                    rationale=item.get("rationale", ""),
                )
            )
        return steps

    def _format_step_summary(self, steps: List[PlanStep]) -> str:
        if not steps:
            return "[none]"
        lines: List[str] = []
        for step in steps:
            lines.append(
                f"{step.step_num}. {step.name} (tool={step.tool_name}) status={step.status} "
                f"condition={step.condition} deps={step.depends_on} result={step.result}"
            )
        return "\n".join(lines)

    def _extract_entity_ids(self, args: Dict[str, Any]) -> List[str]:
        entity_ids: List[str] = []
        patterns = [r"CUST\d+", r"LOAN\d+", r"DOC\d+"]

        def scan(value: Any) -> None:
            if isinstance(value, str):
                for pattern in patterns:
                    entity_ids.extend(re.findall(pattern, value, re.IGNORECASE))
            elif isinstance(value, dict):
                for sub in value.values():
                    scan(sub)
            elif isinstance(value, list):
                for sub in value:
                    scan(sub)

        scan(args)
        return list(dict.fromkeys(entity_ids))

    def _fallback_direct_step(self) -> PlanStep:
        return PlanStep(
            step_num=1,
            name="Direct answer",
            tool_name=None,
            args={},
            depends_on=[],
            condition=None,
            timeout_seconds=60,
            retry_count=0,
            rationale="Fallback direct answer",
        )
