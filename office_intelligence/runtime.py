"""Lazily created application services and shared runtime paths."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agents.core.agent_orchestrator import AgentOrchestrator
    from agents.core.langchain_agent import LangChainAgentExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", PROJECT_ROOT / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_agent: Optional[AgentOrchestrator] = None
_langchain_executor: Optional[LangChainAgentExecutor] = None
_agent_lock = Lock()
_langchain_lock = Lock()


def create_agent() -> AgentOrchestrator:
    from agents.core.agent_orchestrator import AgentOrchestrator

    config = dict(os.environ)
    config["llm_provider"] = os.environ.get("LLM_PROVIDER", "deepseek")
    config["model"] = os.environ.get("DEEPSEEK_MODEL", os.environ.get("LLM_MODEL", "deepseek-chat"))
    config["session_id"] = os.environ.get("SESSION_ID", "backend_api")
    return AgentOrchestrator(config=config)


def get_agent() -> AgentOrchestrator:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = create_agent()
    return _agent


def get_langchain_executor() -> LangChainAgentExecutor:
    from agents.core.langchain_agent import LangChainAgentExecutor

    global _langchain_executor
    if _langchain_executor is None:
        with _langchain_lock:
            if _langchain_executor is None:
                _langchain_executor = LangChainAgentExecutor(config=dict(os.environ))
    return _langchain_executor
