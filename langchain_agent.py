from __future__ import annotations

import json
import os
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

from backend_agent_registry import get_agent_info
from llm_interface import MockLLM


DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", os.environ.get("LLM_MODEL", os.environ.get("OPENAI_MODEL", "deepseek-chat")))
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
DEFAULT_OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_API_BASE")


class LangChainAgentExecutor:
    """A small LangChain runtime for backend agent execution.

    This class is intentionally lightweight: it provides a direct ChatOpenAI
    chat path plus a pandas dataframe agent path for analytics-style agents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.model = config.get("model", DEFAULT_MODEL)
        self.temperature = float(config.get("temperature", DEFAULT_TEMPERATURE))
        self.api_key = config.get("api_key", DEFAULT_OPENAI_API_KEY)
        self.base_url = config.get("base_url", DEFAULT_OPENAI_API_BASE)
        self.llm = self._build_llm()

    def _build_llm(self):
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url

        if not self.api_key and not self.base_url:
            return MockLLM()

        try:
            return ChatOpenAI(**kwargs)
        except Exception:
            return MockLLM()

    def build_system_prompt(self, agent_id: str, prompt: str) -> str:
        agent_info = get_agent_info(agent_id)
        base_message = (
            f"You are the backend execution runtime for the frontend agent '{agent_id}'. "
            "Act as a disciplined, evidence-first AI operator. "
            "Your role is to convert user intent into reliable, business-aware outcomes, "
            "not generic text generation."
        )
        guidelines = (
            "Guidelines: "
            "1. Prioritize the user's business objective and the underlying data context. "
            "2. Stay concise, professional, and action-oriented. "
            "3. State assumptions explicitly when information is missing. "
            "4. Prefer fact-supported answers over speculation. "
            "5. When the request is ambiguous, ask a focused clarifying question rather than guessing. "
            "6. If computation or analysis is required, reason step by step and provide the final output in a clear format. "
            "7. Keep the final answer grounded in actual evidence, not generic advice."
        )
        if agent_info:
            agent_context = (
                f"Agent name: {agent_info['name']}. "
                f"Description: {agent_info['description']}. "
            )
            return agent_context + base_message + " " + guidelines
        return base_message + " " + guidelines

    def chat(self, prompt: str, agent_id: str, max_tokens: Optional[int] = None) -> str:
        messages: List[tuple[str, str]] = [
            ("system", self.build_system_prompt(agent_id, prompt)),
            ("human", prompt),
        ]
        kwargs: Dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self.llm.invoke(messages, **kwargs)
        return getattr(response, "content", "") or str(response)

    def run_dataframe_agent(
        self,
        table_data: Union[str, List[Dict[str, Any]], pd.DataFrame],
        prompt: str,
        agent_type: str = "tool-calling",
        max_iterations: int = 5,
    ) -> str:
        df = self._parse_table_data(table_data)
        agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=df,
            verbose=False,
            agent_type=agent_type,
            allow_dangerous_code=True,
            number_of_head_rows=5,
            max_iterations=max_iterations,
        )
        if hasattr(agent, "invoke"):
            return agent.invoke(prompt)
        return agent.run(prompt)

    def run_langgraph_workflow(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph

        class State(TypedDict):
            result: str

        class Context(TypedDict):
            prompt: str

        def root_node(state: State, runtime):
            return {"result": self.chat(runtime.context["prompt"], agent_id, max_tokens=1024)}

        graph = StateGraph(state_schema=State, context_schema=Context)
        graph.add_node("root", root_node)
        graph.set_entry_point("root")
        graph.set_finish_point("root")
        compiled = graph.compile()
        return compiled.invoke({"result": ""}, context={"prompt": prompt})

    def _parse_table_data(
        self,
        table_data: Union[str, List[Dict[str, Any]], pd.DataFrame],
    ) -> pd.DataFrame:
        if isinstance(table_data, pd.DataFrame):
            return table_data

        if isinstance(table_data, str):
            csv_buffer = StringIO(table_data.strip())
            try:
                return pd.read_csv(csv_buffer)
            except pd.errors.EmptyDataError as exc:
                raise ValueError("CSV data is empty or malformed") from exc
            except Exception as exc:
                # Try JSON fallback for stringified objects
                try:
                    data = json.loads(table_data)
                    return self._parse_table_data(data)
                except Exception:
                    raise ValueError("Unable to parse table data string as CSV or JSON") from exc

        if isinstance(table_data, list):
            if not table_data:
                raise ValueError("Table data list is empty")
            if isinstance(table_data[0], dict):
                return pd.DataFrame(table_data)
            if isinstance(table_data[0], list):
                return pd.DataFrame(table_data)

        raise ValueError(
            "Unsupported table_data format. Use CSV text, a list of dictionaries, or a pandas.DataFrame."
        )
