from __future__ import annotations

import base64
import json
import os
import pathlib
from io import StringIO
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent


FINANCIAL_DATA_PROMPT = """You are an expert financial data analyst.

You have access to price history, volume, and related time series data.
Your goal is to:
- Explore the dataset first (shape, columns, dtypes, missing values)
- Compute key indicators when helpful: moving averages, RSI, Bollinger Bands, volatility, momentum
- Identify trends, support/resistance, and risk signals
- Provide clear recommendations, risk assessments, and actionable observations
- Use visualizations when needed and save them as chart.png
- Think step-by-step: plan, analyze, execute, and summarize

If code execution is required, use the dataframe agent's tool-calling capabilities.
"""

MEMORY_FILE = pathlib.Path("memory_store") / "financial_data_memory.json"
MAX_MEMORY_TURNS = 25
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", os.environ.get("LLM_MODEL", os.environ.get("OPENAI_MODEL", "deepseek-chat")))
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.environ.get("OPENAI_API_BASE") or os.environ.get("DEEPSEEK_API_BASE")
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))


class ConversationMemory:
    def __init__(self, path: pathlib.Path, max_turns: int = MAX_MEMORY_TURNS):
        self.path = path
        self.max_turns = max_turns
        self.history: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        self.history = data[-self.max_turns :]
            except Exception:
                self.history = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.history[-self.max_turns :], fh, indent=2)

    def add_turn(self, user: str, assistant: str) -> None:
        self.history.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "user": user,
                "assistant": assistant,
            }
        )
        self.history = self.history[-self.max_turns :]
        self._save()

    def get_prompt_context(self) -> str:
        if not self.history:
            return ""
        lines = [
            f"User: {entry['user']}\nAssistant: {entry['assistant']}"
            for entry in self.history[-self.max_turns :]
        ]
        return "\n\n".join(lines)


MEMORY = ConversationMemory(MEMORY_FILE)


def _build_llm() -> ChatOpenAI:
    kwargs: Dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
    }
    if DEFAULT_API_KEY:
        kwargs["api_key"] = DEFAULT_API_KEY
    if DEFAULT_API_BASE:
        kwargs["base_url"] = DEFAULT_API_BASE
    return ChatOpenAI(**kwargs)


def _build_prompt(prefix: str, prompt: str) -> str:
    memory_context = MEMORY.get_prompt_context()
    prompt_sections = [prefix]
    if memory_context:
        prompt_sections.append("CONVERSATION MEMORY:\n" + memory_context)
    prompt_sections.append("USER QUESTION:\n" + prompt)
    prompt_sections.append(
        "RESPONSE FORMAT:\nProvide a concise executive summary. When using code, include only the code block and ensure visualizations are saved as chart.png."
    )
    return "\n\n".join(prompt_sections)


def _create_financial_dataframe_agent(df: pd.DataFrame) -> Any:
    llm = _build_llm()
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=False,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        number_of_head_rows=10,
        prefix=FINANCIAL_DATA_PROMPT,
        max_iterations=10,
    )


def _parse_csv(csv_content: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(csv_content))


def _encode_chart() -> Optional[str]:
    chart_path = os.path.abspath("chart.png")
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    return None


def run_financial_analyst(csv_content: Optional[str], table_json: Optional[List[Dict[str, Any]]], prompt: str) -> Dict[str, Any]:
    if not prompt or not prompt.strip():
        return {"error": "Prompt is required for financial-data."}

    if csv_content:
        try:
            df = _parse_csv(csv_content)
        except Exception as exc:
            return {"error": f"CSV parsing failed: {exc}"}
    elif table_json is not None:
        try:
            df = pd.DataFrame(table_json)
        except Exception as exc:
            return {"error": f"Table JSON parsing failed: {exc}"}
    else:
        return {"error": "financial-data requires CSV input or table JSON input."}

    if df.empty:
        return {"error": "No financial data provided."}

    agent = _create_financial_dataframe_agent(df)
    try:
        if hasattr(agent, "invoke"):
            answer = agent.invoke(prompt)
        else:
            answer = agent.run(prompt)
    except Exception as exc:
        return {"error": f"Agent execution failed: {exc}"}

    chart_png_base64 = _encode_chart()
    MEMORY.add_turn(prompt, answer)

    return {
        "answer": answer,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "chart_created": chart_png_base64 is not None,
        "chart_png_base64": chart_png_base64,
    }
