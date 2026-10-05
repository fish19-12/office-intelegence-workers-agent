"""
Agent Orchestrator Module

Wires all service classes together (document management, semantic search, memory,
planning, tool execution) into a single unified interface for frontend clients.

Replaces EnterpriseFileAgent as the primary object that user-facing code interacts with.

Example:
    >>> from agent_orchestrator import AgentOrchestrator
    >>> config = {"llm_provider": "deepseek", "SUPABASE_URL": "..."}
    >>> agent = AgentOrchestrator(config)
    >>> asset = agent.ingest_file("report.pdf")
    >>> preview = agent.plan("Generate quarterly analysis")
    >>> result = agent.execute_plan(preview, confirmed=True)
    >>> print(result.answer)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from context_retriever import ContextRetriever, RetrievedChunk
from document_manager import DocumentAsset, DocumentManager
from embeddings_rag import AdvancedRAG
from llm_interface import LLMFactory
from mcp_manager import MCPManager
from memory_manager import MemoryManager
from reactive_planner import PlanResult, ReactivePlanner
from report_builder import ReportBuilder, ReportResult as BuiltReportResult
from supabase_client import SupabaseClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ConfirmationRequiredError(Exception):
    """Raised when plan contains irreversible tools requiring user confirmation."""
    
    def __init__(self, irreversible_tools: List[str]):
        self.irreversible_tools = irreversible_tools
        msg = (
            f"This plan uses irreversible tools that cannot be undone: "
            f"{', '.join(irreversible_tools)}. "
            "Please confirm before proceeding."
        )
        super().__init__(msg)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class QueryResult:
    """Result of a semantic query."""
    
    answer: str
    chunks: List[RetrievedChunk]
    query: str


@dataclass
class PlanPreview:
    """Preview of planned steps before execution (dry-run)."""
    
    goal: str
    planned_tools: List[str]
    irreversible_tools: List[str]
    needs_confirmation: bool
    chunks: List[RetrievedChunk]


# ReportResult is now imported from report_builder module
# Keeping this alias for backward compatibility with simple report generation
# For advanced reports with code execution, use report_builder.ReportResult directly





# ============================================================================
# AGENT ORCHESTRATOR
# ============================================================================

class AgentOrchestrator:
    """
    Central orchestrator that wires all service classes together.
    
    Provides a unified interface for:
    - Document ingestion (DocumentManager)
    - Semantic search (ContextRetriever)
    - Session memory (MemoryManager)
    - Plan generation and execution (ReactivePlanner)
    - Report generation (SimpleReportBuilder)
    
    Replaces EnterpriseFileAgent as the primary frontend interface.
    
    Attributes:
        config: Configuration dict with LLM provider, Supabase URL, etc.
        llm: Language model instance
        rag: Embeddings RAG for local vector search
        supabase: Optional Supabase client for remote sync
        mcp: Tool registry and execution manager
        documents: Document ingestion service
        retriever: Semantic search service
        memory: Session memory service
        planner: Reactive planning service
        reporter: Report generation service
    """
    
    # Tools that perform irreversible actions (require user confirmation)
    IRREVERSIBLE_TOOLS = {
        "email_send",
        "payment_reminder",
        "client_onboarding",
        "slack",
        "teams",
        "google_drive",
        "google_sheets",
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AgentOrchestrator with all service classes.
        
        Args:
            config: Configuration dict with keys:
                - llm_provider: "deepseek", "openai", "huggingface" (default: "deepseek")
                - model: model name
                - SUPABASE_URL: optional Supabase URL
                - SUPABASE_KEY: optional Supabase API key
                - session_id: optional session identifier
                - Any other LLM-specific config
        """
        config = config or {}
        self.config = config
        
        logger.info("Initializing AgentOrchestrator")
        
        # Initialize LLM
        llm_provider = config.get("llm_provider", "deepseek")
        logger.info(f"Creating LLM: {llm_provider}")
        self.llm = LLMFactory.create(llm_provider, config)
        
        # Initialize RAG (local embeddings)
        logger.info("Creating AdvancedRAG")
        # AdvancedRAG expects an LLM adapter for decomposition and self-rag decisions
        self.rag = AdvancedRAG(llm=self.llm)
        
        # Initialize Supabase (optional)
        if config.get("SUPABASE_URL"):
            logger.info("Creating SupabaseClient")
            try:
                self.supabase = SupabaseClient(
                    url=config.get("SUPABASE_URL"),
                    key=config.get("SUPABASE_KEY"),
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase: {e}")
                self.supabase = None
        else:
            self.supabase = None
        
        # Initialize MCP (tool registry)
        logger.info("Creating MCPManager")
        self.mcp = MCPManager(llm=self.llm)
        
        # Initialize service classes
        logger.info("Creating DocumentManager")
        self.documents = DocumentManager(self.rag, self.supabase)
        
        logger.info("Creating ContextRetriever")
        self.retriever = ContextRetriever(self.rag, self.supabase)
        
        logger.info("Creating MemoryManager")
        session_id = config.get("session_id")
        # MemoryManager now expects llm, optional persist_directory, and session_id
        self.memory = MemoryManager(
            self.llm,
            persist_directory="memory_store",
            session_id=session_id
        )
        
        logger.info("Creating ReactivePlanner")
        self.planner = ReactivePlanner(self.llm, self.mcp, self.memory, self.retriever)
        
        logger.info("Creating ReportBuilder")
        self.reporter = ReportBuilder(self.llm, self.mcp, self.retriever)
        
        logger.info("AgentOrchestrator initialized successfully")
    
    @property
    def active_model(self) -> str:
        """Model identifier for frontend status display."""
        class_name = self.llm.__class__.__name__
        model_name = self.config.get("model", "unknown")
        return f"{class_name} · {model_name}"
    
    @property
    def active_model_status(self) -> str:
        """LLM status: production | fallback | mock."""
        llm_class = self.llm.__class__.__name__
        
        if "Mock" in llm_class:
            return "mock"
        elif "Fallback" in llm_class or "Error" in llm_class:
            return "fallback"
        else:
            return "production"
    
    def ingest_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentAsset:
        """
        Ingest a document file.
        
        Thin delegation to DocumentManager with memory recording.
        
        Args:
            file_path: Path to file to ingest
            metadata: Optional metadata (category, priority, user_goal, etc.)
        
        Returns:
            DocumentAsset with doc_id, chunk_count, and metadata
        """
        logger.info(f"Ingesting file: {file_path}")
        
        asset = self.documents.ingest(file_path, metadata)
        
        # Record in memory
        self.memory.record(
            "document_ingested",
            f"Ingested {asset.filename} ({asset.chunk_count} chunks)",
            detail={"doc_id": asset.doc_id, "filename": asset.filename},
            entity_ids=[asset.doc_id],
        )
        
        logger.info(f"File ingested: {asset.doc_id} with {asset.chunk_count} chunks")
        return asset
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, str]] = None,
    ) -> QueryResult:
        """
        Answer a semantic query using retrieved documents.
        
        Steps:
        1. Retrieve relevant chunks
        2. Format as context
        3. Generate answer using LLM
        4. Record in memory
        
        Args:
            query: Question or search query
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters
        
        Returns:
            QueryResult with answer and retrieved chunks
        """
        logger.info(f"Processing query: {query[:60]}...")
        
        # Step 1: Retrieve chunks
        chunks = self.retriever.retrieve(query, top_k=top_k, filters=filters)
        logger.debug(f"Retrieved {len(chunks)} chunks")
        
        # Step 2: Format context
        context_str = self.retriever.format_for_prompt(chunks)
        memory_str = self.memory.get_prompt_context()
        
        # Step 3: Generate answer
        answer_prompt = (
            "Answer this query using only the provided context. "
            "If the context does not contain the answer, say 'This information is not available in the provided documents.'\n\n"
            f"Memory:\n{memory_str}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Query: {query}\n\n"
            "Answer:"
        )
        
        try:
            answer = self.llm.generate(answer_prompt)
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            answer = f"Error generating answer: {str(e)}"
        
        # Step 4: Record in memory
        self.memory.record(
            "query_answered",
            f"Query: {query[:60]}...",
            detail={"query": query, "top_k": top_k, "chunks_retrieved": len(chunks)},
        )
        
        logger.info(f"Query answered using {len(chunks)} chunks")
        return QueryResult(answer=answer, chunks=chunks, query=query)
    
    def plan(self, goal: str, top_k: int = 5) -> PlanPreview:
        """
        Generate a preview of planned steps (dry-run, not executed).
        
        Steps:
        1. Retrieve context for goal
        2. Ask LLM to suggest tools (without executing)
        3. Detect irreversible tools
        4. Return preview for user confirmation if needed
        
        Args:
            goal: User's goal/request
            top_k: Number of chunks to retrieve
        
        Returns:
            PlanPreview with planned tools and confirmation requirement
        """
        logger.info(f"Planning for goal: {goal[:60]}...")
        
        # Step 1: Retrieve context
        chunks = self.retriever.retrieve_for_goal(goal, top_k=top_k)
        logger.debug(f"Retrieved {len(chunks)} chunks for planning")
        
        # Step 2: Dry-run plan to detect tools
        tool_schemas = self.planner._format_tool_schemas()
        planning_prompt = (
            f"Goal: {goal}\n\n"
            f"Available tools:\n{tool_schemas}\n\n"
            "Which tools from this list would you use to achieve this goal? "
            "Reply ONLY with a JSON array of tool names, e.g. [\"tool_a\", \"tool_b\"]. "
            "If no tools needed, reply with []."
        )
        
        try:
            raw_response = self.llm.generate(planning_prompt, max_tokens=200)
            # Extract the first JSON array-looking substring robustly (handles trailing explanation)
            import re
            match = re.search(r"\[.*?\]", raw_response, re.S)
            if match:
                try:
                    planned_tools = json.loads(match.group())
                except json.JSONDecodeError:
                    planned_tools = []
            else:
                planned_tools = []
        except Exception as e:
            logger.warning(f"Failed to parse planned tools: {e}")
            planned_tools = []
        
        logger.debug(f"Planned tools: {planned_tools}")
        
        # Step 3: Detect irreversible tools
        irreversible_planned = [
            t for t in planned_tools
            if any(ir in t.lower() for ir in self.IRREVERSIBLE_TOOLS)
        ]
        needs_confirmation = len(irreversible_planned) > 0
        
        logger.info(
            f"Plan preview: {len(planned_tools)} tools, "
            f"{len(irreversible_planned)} irreversible"
        )
        
        return PlanPreview(
            goal=goal,
            planned_tools=planned_tools,
            irreversible_tools=irreversible_planned,
            needs_confirmation=needs_confirmation,
            chunks=chunks,
        )
    
    def _get_react_tool_schemas(self) -> List[dict]:
        """Return only the requested ReAct tools for the OpenAI tool-aware prompt."""
        desired_tools = [
            "email_send",
            "google_sheets_append",
            "word_report",
            "slack_post",
            "google_drive_upload",
            "loan_report_drive",
        ]
        return self.mcp.get_tool_schemas(desired_tools)

    def _format_scratchpad(self, scratchpad: List[str]) -> str:
        if not scratchpad:
            return "None"
        return "\n".join(f"- {line}" for line in scratchpad)

    def _build_react_prompt(
        self,
        goal: str,
        context: str,
        memory_context: str,
        scratchpad: List[str],
    ) -> str:
        return (
            "You are a reasoning agent for a microfinance workflow. "
            "Your job is to complete the user's goal by thinking, selecting the best available tool, "
            "executing it, observing the result, and then thinking again until the goal is complete.\n\n"
            "If you decide to use a tool, respond with exactly:\n"
            "ACTION: <tool_name> {\"arg\": \"value\"} \n"
            "Do not add any extra text before the ACTION line.\n"
            "If you do not need a tool and the task is finished, answer directly with the final result.\n\n"
            f"Goal:\n{goal}\n\n"
            f"Context:\n{context}\n\n"
            f"Memory:\n{memory_context}\n\n"
            f"Scratchpad:\n{self._format_scratchpad(scratchpad)}\n"
        )

    def _run_reactive_goal(
        self,
        goal: str,
        context_chunks: List[RetrievedChunk],
        max_iterations: int = 10,
    ) -> PlanResult:
        """Run a ReAct loop to complete the goal using tool calls selected by the LLM."""
        start_time = time.time()
        context = self.retriever.format_for_prompt(context_chunks)
        memory_ctx = self.memory.get_prompt_context()
        tools = self._get_react_tool_schemas()

        conversation: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful microfinance agent. Use available tools only when they help achieve the goal. "
                    "If a tool fails, observe the error and choose another approach."
                ),
            }
        ]

        scratchpad: List[str] = []
        executed_steps: List[PlanStep] = []
        final_answer: Optional[str] = None

        for iteration in range(1, max_iterations + 1):
            logger.info(f"ReAct iteration {iteration} for goal: {goal[:60]}...")
            prompt = self._build_react_prompt(goal, context, memory_ctx, scratchpad)
            response = self.llm.generate_with_tools(prompt, tools=tools, messages=conversation)

            if response.type == "text":
                final_answer = response.text.strip()
                scratchpad.append(f"Thought {iteration}: direct answer produced.")
                logger.info("LLM provided a final answer without invoking a tool.")
                break

            tool_name = response.tool_name
            tool_args = response.tool_args or {}
            if not tool_name:
                final_answer = response.text.strip() or "No tool selected by the LLM."
                scratchpad.append(f"Thought {iteration}: no valid tool selected.")
                logger.warning("LLM did not select a tool. Ending loop with direct answer.")
                break

            step = PlanStep(
                step_num=iteration,
                tool_name=tool_name,
                args=tool_args,
                rationale="LLM selected a tool in the ReAct loop.",
            )
            scratchpad.append(f"Thought {iteration}: choose {tool_name} with args {tool_args}")

            conversation.append({
                "role": "assistant",
                "content": f"ACTION: {tool_name} {json.dumps(tool_args)}",
            })

            try:
                start = time.time()
                result = self.mcp.call_tool(tool_name, **tool_args)
                step.result = result
                step.status = "success"
                step.duration_ms = int((time.time() - start) * 1000)
                executed_steps.append(step)
                self.memory.record(
                    "tool_executed",
                    f"{tool_name} executed in ReAct iteration {iteration}",
                    detail=result,
                )

                observation = json.dumps(result, default=str)
                scratchpad.append(f"Observation {iteration}: {observation}")
                conversation.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": observation,
                })
            except Exception as exc:
                step.status = "error"
                step.error_msg = str(exc)
                executed_steps.append(step)
                self.memory.record(
                    "tool_error",
                    f"{tool_name} failed in ReAct iteration {iteration}",
                    detail={"error": str(exc)},
                )
                error_text = json.dumps({"error": str(exc)}, default=str)
                scratchpad.append(f"Observation {iteration}: tool failed: {str(exc)}")
                conversation.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": error_text,
                })
                logger.warning(f"Tool {tool_name} failed: {exc}. Continuing ReAct loop.")
                continue

        if final_answer is None:
            final_answer = (
                "Reached the maximum ReAct iterations without a direct final answer. "
                "Review the executed steps for partial progress."
            )

        plan_result = PlanResult(
            answer=final_answer,
            steps=executed_steps,
            partial=len(executed_steps) >= max_iterations,
            goal_achieved=(final_answer is not None and not final_answer.startswith("Reached the maximum")),
        )
        plan_result.total_duration_ms = int((time.time() - start_time) * 1000)
        return plan_result

    def execute_plan(
        self,
        plan_preview: PlanPreview,
        confirmed: bool = False,
    ) -> PlanResult:
        """
        Execute a planned workflow.
        
        If plan contains irreversible tools and not confirmed, raises ConfirmationRequiredError.
        Otherwise, runs the full ReAct reasoning loop for the requested goal.
        
        Args:
            plan_preview: PlanPreview from plan() method
            confirmed: Whether user has confirmed irreversible actions
        
        Returns:
            PlanResult with answer, steps, and goal_achieved flag
        
        Raises:
            ConfirmationRequiredError: If irreversible tools and not confirmed
        """
        logger.info(f"Executing plan for goal: {plan_preview.goal[:60]}...")
        
        if plan_preview.needs_confirmation and not confirmed:
            logger.warning("Confirmation required for irreversible tools")
            raise ConfirmationRequiredError(plan_preview.irreversible_tools)
        
        result = self._run_reactive_goal(plan_preview.goal, plan_preview.chunks)
        
        status = "achieved" if result.goal_achieved else "partial"
        self.memory.record(
            "workflow_completed",
            f"Goal: {plan_preview.goal[:60]}... — {status}",
            detail={
                "goal": plan_preview.goal,
                "achieved": result.goal_achieved,
                "steps": len(result.steps),
            },
        )
        
        logger.info(f"Plan executed: {status} (duration: {result.total_duration_ms}ms)")
        return result
    
    def generate_report(
        self,
        goal: str,
        query: Optional[str] = None,
        top_k: int = 5,
    ) -> BuiltReportResult:
        """
        Generate a formatted report with optional code execution.
        
        Returns a ReportResult with sections and pending code blocks.
        Code blocks must be approved and executed separately via execute_code_block().
        After all code blocks are resolved, call finalize() to generate HTML.
        
        Args:
            goal: Report goal/title
            query: Optional search query (defaults to goal)
            top_k: Number of chunks to retrieve
        
        Returns:
            BuiltReportResult with sections and code blocks
        """
        logger.info(f"Generating report for goal: {goal[:60]}...")
        
        search_query = query or goal
        chunks = self.retriever.retrieve(search_query, top_k=top_k)
        
        result = self.reporter.build(goal, search_query, chunks)
        
        # Record in memory
        self.memory.record(
            "report_generated",
            f"Report: {goal[:60]}...",
            detail={
                "goal": goal,
                "chunks": len(chunks),
                "sections": len(result.sections),
                "pending_code_blocks": len(result.pending_code_blocks),
            },
        )
        
        logger.info(
            f"Report generated: {len(result.sections)} sections, "
            f"{len(result.pending_code_blocks)} pending code blocks"
        )
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get full system status for Admin Diagnostics tab.
        
        Returns:
            Dict with status of all system components
        """
        try:
            doc_categories = list(set(
                d.category for d in self.documents.list_documents()
            ))
        except Exception:
            doc_categories = []
        
        status = {
            "llm": {
                "model": self.active_model,
                "status": self.active_model_status,
            },
            "rag": self.retriever.sync_status,
            "memory": self.memory.get_full_context(),
            "tools": self.mcp.registry.list(),
            "documents": {
                "count": len(self.documents.documents),
                "categories": doc_categories,
            },
        }
        
        return status
