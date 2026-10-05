"""Specialist agents: lightweight facade that re-exports individual agent modules.

This file keeps the public API stable (imports in other modules) while each
specialist lives in its own file (e.g. `data_agent.py`).
"""
from __future__ import annotations

from importlib import import_module

# Import specific agent classes from their new modules and expose them here so
# existing imports like `from specialist_agents import DataAgent` continue to
# work.
from data_agent import DataAgent  # type: ignore
from report_agent import ReportAgent  # type: ignore
from communication_agent import CommunicationAgent  # type: ignore
from risk_agent import RiskAgent  # type: ignore
from search_agent import SearchAgent  # type: ignore

__all__ = [
    "DataAgent",
    "ReportAgent",
    "CommunicationAgent",
    "RiskAgent",
    "SearchAgent",
]
