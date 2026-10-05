import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.core.langchain_agent import LangChainAgentExecutor


def test_parse_csv_table_data():
    executor = LangChainAgentExecutor(config={"api_key": "test"})
    df = executor._parse_table_data("a,b\n1,2\n3,4\n")

    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 1] == 4


def test_run_langgraph_workflow(monkeypatch):
    executor = LangChainAgentExecutor(config={"api_key": "test"})

    monkeypatch.setattr(executor, "chat", lambda prompt, agent_id, max_tokens=None: "graph answer")
    result = executor.run_langgraph_workflow("test-agent", "hello world")

    assert result["result"] == "graph answer"


def test_parse_json_table_data():
    executor = LangChainAgentExecutor(config={"api_key": "test"})
    df = executor._parse_table_data([{"a": 1, "b": 2}, {"a": 3, "b": 4}])

    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]
    assert df.iloc[0, 1] == 2
    assert df.iloc[1, 0] == 3
