"""Specialist agents: lightweight facade that re-exports individual agent modules.

This file keeps the public API stable (imports in other modules) while each
specialist lives in its own file (e.g. `agents.core.data_agent.py`).
"""
from __future__ import annotations

# Import specific agent classes from their new modules and expose them here so
# existing imports like `from agents.core.specialist_agents import DataAgent` continue to
# work.
from agents.core.data_agent import DataAgent  # type: ignore
from agents.core.report_agent import ReportAgent  # type: ignore
from agents.core.communication_agent import CommunicationAgent  # type: ignore
from agents.core.risk_agent import RiskAgent  # type: ignore
from agents.core.search_agent import SearchAgent  # type: ignore

__all__ = [
    "DataAgent",
    "ReportAgent",
    "CommunicationAgent",
    "RiskAgent",
    "SearchAgent",
]
