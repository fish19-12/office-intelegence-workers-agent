from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

from llm_interface import BaseLLM, LLMFactory, MockLLM, ToolCallResponse
from tools import PythonExecutionTool

SQL_ANALYST_SYSTEM_PROMPT = """You are an Advanced AI Data Analyst Agent.

You have access to:

DATABASE TOOLS
--------------
- SQLite
- PostgreSQL
- MySQL
- SQLAlchemy
- Database schema inspection
- SQL execution

DATA ANALYSIS TOOLS
-------------------
- Pandas
- NumPy
- SciPy
- Polars

VISUALIZATION TOOLS
-------------------
- Matplotlib
- Plotly

MACHINE LEARNING TOOLS
----------------------
- Scikit-learn
- XGBoost
- LightGBM

FILE TOOLS
----------
- CSV
- Excel
- JSON
- SQLite databases
- Parquet
- Text files

PYTHON EXECUTION
----------------
You can write and execute Python code whenever needed.

CONVERSATIONAL MEMORY
---------------------
You have access to the previous 25 conversation turns.

Use memory to:

- Remember user goals
- Remember uploaded files
- Remember previous analysis
- Remember previous reports
- Remember business context
- Continue unfinished work

If a user references previous work, use memory before asking questions.

------------------------------------------------

PRIMARY RESPONSIBILITIES

1. Understand the user's request.

2. Determine which uploaded files or databases are relevant.

3. Inspect data automatically.

4. Discover schema and relationships.

5. Generate SQL when necessary.

6. Generate Python code when necessary.

7. Execute tools.

8. Analyze results.

9. Create visualizations when helpful.

10. Generate insights and recommendations.

11. Continue reasoning until the task is solved.

------------------------------------------------

DATABASE ANALYSIS

For databases:

- List tables
- Identify columns
- Identify primary keys
- Identify foreign keys
- Identify indexes
- Calculate row counts

When needed:

- Generate SQL
- Execute SQL
- Analyze results

Never require the user to write SQL.

------------------------------------------------

PYTHON ANALYSIS

When SQL alone is insufficient:

- Write Python code
- Execute code
- Use Pandas
- Use NumPy
- Use statistical libraries

Perform:

- Aggregation
- Forecasting
- Trend analysis
- Classification
- Clustering
- Correlation analysis
- Anomaly detection

------------------------------------------------

VISUALIZATION

When useful:

Generate:

- Line charts
- Bar charts
- Pie charts
- Histograms
- Scatter plots
- Heatmaps

Explain what the chart shows.

------------------------------------------------

REPORTING

Always produce:

1. Executive Summary
2. Key Findings
3. Supporting Statistics
4. Insights
5. Recommendations

------------------------------------------------

MEMORY UTILIZATION

You remember the previous 25 conversation turns.

Use memory to:

- Continue analyses
- Reference prior uploads
- Track ongoing projects
- Avoid asking repetitive questions

If the user uploads a newer version of a file:

- Compare against previous versions
- Highlight differences
- Update findings

------------------------------------------------

DATABASE ANALYSIS

For databases:

- List tables
- Identify columns
- Identify primary keys
- Identify foreign keys
- Identify indexes
- Calculate row counts

When needed:

- Generate SQL
- Execute SQL
- Analyze results

Never require the user to write SQL.

------------------------------------------------

PYTHON ANALYSIS

When SQL alone is insufficient:

- Write Python code
- Execute code
- Use Pandas
- Use NumPy
- Use statistical libraries

Perform:

- Aggregation
- Forecasting
- Trend analysis
- Classification
- Clustering
- Correlation analysis
- Anomaly detection

------------------------------------------------

VISUALIZATION

When useful:

Generate:

- Line charts
- Bar charts
- Pie charts
- Histograms
- Scatter plots
- Heatmaps

Explain what the chart shows.

------------------------------------------------

REPORTING

Always produce:

1. Executive Summary
2. Key Findings
3. Supporting Statistics
4. Insights
5. Recommendations

------------------------------------------------

MEMORY UTILIZATION

You remember the previous 25 conversation turns.

Use memory to:

- Continue analyses
- Reference prior uploads
- Track ongoing projects
- Avoid asking repetitive questions

If the user uploads a newer version of a file:

- Compare against previous versions
- Highlight differences
- Update findings

------------------------------------------------

CODE EXECUTION STRATEGY

Before executing code:
1. Understand the problem
2. Determine required tools
3. Generate code
4. Execute code
5. Analyze output
6. Continue reasoning if needed
7. Return final answer

Do not stop after generating code.

Always analyze the results.

------------------------------------------------

SAFETY

Only perform safe operations.

For databases:

Allowed:
- SELECT
- WITH
- PRAGMA
- EXPLAIN

Disallowed:
- DROP
- DELETE
- UPDATE
- INSERT
- ALTER
- TRUNCATE

Never modify user data unless explicitly allowed.

------------------------------------------------

You are not a SQL executor.

You are a complete AI Data Analyst capable of database analysis, file analysis, statistical analysis, machine learning, visualization, business intelligence, and report generation.
"""

SQL_AGENT_PROMPT = """You are a self-sufficient SQL analyst. Use the variables available in python execution.

If you are given table JSON, df is available as a pandas DataFrame.
If you are given a SQLite database file, db_path is available as the path to the database and conn is a sqlite3 connection.

If analysis needs code, return a single ```python``` code block only. Do not include additional markdown around the code.

If code is required, keep the output minimal and let the execution environment capture stdout and files.

If you produce visualizations, save them as chart.png and include a summary of the chart in the final answer.

Always think step-by-step and write code only when it is required to answer the user.
"""

MEMORY_FILE = pathlib.Path("memory_store") / "sql_analyst_memory.json"
MAX_MEMORY_TURNS = 25

DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", os.environ.get("LLM_MODEL", os.environ.get("OPENAI_MODEL", "deepseek-chat")))
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.environ.get("OPENAI_API_BASE") or os.environ.get("DEEPSEEK_API_BASE")
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

LLM_CONFIG = dict(os.environ)
LLM_PROVIDER = LLM_CONFIG.get("LLM_PROVIDER", LLM_CONFIG.get("llm_provider", "deepseek"))
TOOL_LLM = LLMFactory.create(LLM_PROVIDER, LLM_CONFIG)
PYTHON_TOOL = PythonExecutionTool()
AVAILABLE_TOOLS = [PYTHON_TOOL.to_openai_tool()]


def _build_langchain_llm():
    kwargs: Dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
    }
    if DEFAULT_API_KEY:
        kwargs["api_key"] = DEFAULT_API_KEY
    if DEFAULT_API_BASE:
        kwargs["base_url"] = DEFAULT_API_BASE

    if not DEFAULT_API_KEY and not DEFAULT_API_BASE:
        return MockLLM()

    try:
        return ChatOpenAI(**kwargs)
    except Exception:
        return MockLLM()


def _run_pandas_dataframe_agent(table_json: List[Dict[str, Any]], prompt: str, max_iterations: int = 5) -> str:
    df = pd.DataFrame(table_json)
    llm = _build_langchain_llm()
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=False,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        number_of_head_rows=5,
        prefix=SQL_AGENT_PROMPT,
        max_iterations=max_iterations,
    )
    if hasattr(agent, "invoke"):
        return agent.invoke(prompt)
    return agent.run(prompt)


def _run_db_file_langchain(db_path: str, prompt: str) -> str:
    schema = _analyze_db_schema(db_path)
    description_lines: List[str] = [
        f"Database has {len(schema['tables'])} tables."
    ]
    for table in schema["tables"]:
        column_names = ", ".join([col["name"] for col in table["columns"]])
        description_lines.append(
            f"Table '{table['table']}' has {table['row_count']} rows and columns: {column_names}."
        )
        if table["preview"]:
            preview_values = table["preview"][:3]
            description_lines.append(f"Sample rows for '{table['table']}': {preview_values}")
    schema_prompt = "\n".join(description_lines)
    llm = _build_langchain_llm()
    messages = [
        ("system", "You are an expert SQL and SQLite database analyst. Use the database schema and sample data to analyze the database and answer the user's question."),
        ("user", f"Database summary:\n{schema_prompt}\n\nUser question:\n{prompt}")
    ]
    response = llm.invoke(messages)
    return getattr(response, "content", "") or str(response)


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


def _connect_sqlite(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    return sqlite3.connect(db_path)


def _analyze_db_schema(db_path: str) -> Dict[str, Any]:
    conn = _connect_sqlite(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        details: List[Dict[str, Any]] = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info('{table}')")
            cols = [
                dict(
                    zip(
                        ["cid", "name", "type", "notnull", "dflt_value", "pk"],
                        row,
                    )
                )
                for row in cursor.fetchall()
            ]
            cursor.execute(f"SELECT COUNT(1) FROM '{table}'")
            row_count = cursor.fetchone()[0]
            cursor.execute(f"SELECT * FROM '{table}' LIMIT 5")
            preview = [
                dict(zip([desc[0] for desc in cursor.description], row))
                for row in cursor.fetchall()
            ]
            details.append(
                {
                    "table": table,
                    "columns": cols,
                    "row_count": row_count,
                    "preview": preview,
                }
            )
        return {"tables": details}
    finally:
        conn.close()


def analyze_db_file(db_path: str) -> Dict[str, Any]:
    return _analyze_db_schema(db_path)


def run_sql_query(db_path: str, query: str) -> Dict[str, Any]:
    conn = _connect_sqlite(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        preview = df.head(10).to_dict(orient="records")
        stats = df.describe(include="all").to_dict()
        return {"rows": len(df), "preview": preview, "stats": stats}
    finally:
        conn.close()


def run_sql_query_on_table(table_json: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    df = pd.DataFrame(table_json)
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql("data", conn, index=False, if_exists="replace")
        result = pd.read_sql_query(query, conn)
        preview = result.head(10).to_dict(orient="records")
        stats = result.describe(include="all").to_dict()
        return {"rows": len(result), "preview": preview, "stats": stats}
    finally:
        conn.close()


def _dump_table_json_to_csv(table_json: List[Dict[str, Any]], work_dir: pathlib.Path) -> str:
    df = pd.DataFrame(table_json)
    csv_path = work_dir / "table_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def _build_data_summary(db_path: Optional[str], table_json: Optional[List[Dict[str, Any]]]) -> str:
    if db_path:
        schema = _analyze_db_schema(db_path)
        summary_lines = [f"Database has {len(schema['tables'])} tables."]
        for table in schema["tables"]:
            summary_lines.append(
                f"Table {table['table']} has {table['row_count']} rows and {len(table['columns'])} columns."
            )
        return "\n".join(summary_lines)
    if table_json is not None:
        df = pd.DataFrame(table_json)
        return (
            f"Table JSON contains {len(df)} rows and {len(df.columns)} columns. "
            f"Columns: {', '.join(map(str, df.columns.tolist()))}."
        )
    return "No data source provided."


def _build_prompt(prompt: str, db_path: Optional[str], table_json: Optional[List[Dict[str, Any]]]) -> str:
    memory_context = MEMORY.get_prompt_context()
    data_summary = _build_data_summary(db_path, table_json)
    prompt_sections = [SQL_ANALYST_SYSTEM_PROMPT]

    if memory_context:
        prompt_sections.append("CONVERSATION MEMORY:\n" + memory_context)
    prompt_sections.append("DATA SUMMARY:\n" + data_summary)
    prompt_sections.append("INSTRUCTIONS:\n" + SQL_AGENT_PROMPT)
    prompt_sections.append("USER QUESTION:\n" + prompt)
    prompt_sections.append(
        "RESPONSE FORMAT:\nIf code is required, return a single ```python``` block only. "
        "If code is not necessary, answer directly."
    )
    return "\n\n".join(prompt_sections)


def _extract_python_code(text: str) -> Optional[str]:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        return fenced.group(1).strip()
    return None


def _clean_analysis(text: str, code: Optional[str]) -> str:
    if not code:
        return text.strip()
    code_block = f"```python\n{code}\n```"
    return text.replace(code_block, "").strip()


def _execute_python_code(
    code: str,
    db_path: Optional[str] = None,
    table_json: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    work_dir = pathlib.Path(tempfile.mkdtemp(prefix="sql_analyst_"))
    data_csv_path = None
    copied_db_path = None

    if db_path:
        copied_db_path = str(work_dir / "data.db")
        shutil.copy(db_path, copied_db_path)
    if table_json is not None:
        data_csv_path = _dump_table_json_to_csv(table_json, work_dir)

    script_path = work_dir / "analysis.py"
    wrapped_code = textwrap.dedent(
        f"""
        import json
        import os
        import sqlite3
        import sys
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')

        db_path = {json.dumps(copied_db_path)}
        table_csv = {json.dumps(data_csv_path)}
        conn = sqlite3.connect(db_path) if db_path else None
        df = pd.read_csv(table_csv) if table_csv else None

        try:
{code}
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        """
    )
    script_path.write_text(wrapped_code, encoding="utf-8")

    process = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=180,
    )

    chart_path = work_dir / "chart.png"
    chart_created = chart_path.exists()
    chart_png_base64 = None
    if chart_created:
        import base64

        with open(chart_path, "rb") as fh:
            chart_png_base64 = base64.b64encode(fh.read()).decode("utf-8")

    return {
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "work_dir": str(work_dir),
        "chart_created": chart_created,
        "chart_png_base64": chart_png_base64,
    }


def _execute_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == PYTHON_TOOL.name:
        code = tool_args.get("code", "")
        if not code:
            raise ValueError("python_execute requires a 'code' argument.")
        return PYTHON_TOOL.execute(code=code, timeout=180)
    raise ValueError(f"Unsupported tool: {tool_name}")


def _build_tool_messages(prompt: str, db_path: Optional[str], table_json: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    system_prompt = _build_prompt(prompt, db_path, table_json)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def agent_reasoning_loop(
    prompt: str,
    db_path: Optional[str] = None,
    table_json: Optional[List[Dict[str, Any]]] = None,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Iterative ReAct-style reasoning loop for SQL analysis.
    
    Supports multi-turn tool use with schema inspection, SQL execution,
    and additional analysis up to max_iterations rounds.
    
    Args:
        prompt: Initial user question
        db_path: Path to SQLite database file
        table_json: Table data as list of dicts
        max_iterations: Max reasoning iterations (default 5)
    
    Returns:
        Dict with final analysis, code, execution details, and reasoning trace
    """
    if not prompt or not prompt.strip():
        return {"error": "Prompt cannot be empty."}
    
    # Initialize messages with system prompt and user query
    messages = _build_tool_messages(prompt, db_path, table_json)
    
    # Track reasoning steps for transparency
    scratchpad: List[str] = []
    executed_tools: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None
    
    for iteration in range(1, max_iterations + 1):
        # Query LLM for next action (tool or text response)
        response = TOOL_LLM.generate_with_tools(
            prompt=prompt if iteration == 1 else "Continue analysis or provide final answer based on tool results.",
            tools=AVAILABLE_TOOLS,
            messages=messages,
        )
        
        # If LLM returns text directly, use as final answer
        if response.type == "text":
            final_answer = response.text.strip()
            scratchpad.append(f"Thought {iteration}: Direct response produced final answer.")
            break
        
        # LLM wants to use a tool
        tool_name = response.tool_name
        tool_args = response.tool_args or {}
        
        if not tool_name:
            final_answer = response.text.strip() or "No tool selected by the LLM."
            scratchpad.append(f"Thought {iteration}: No valid tool selected.")
            break
        
        scratchpad.append(f"Thought {iteration}: Invoking {tool_name} to analyze data.")
        
        # Execute tool
        try:
            tool_result = _execute_tool_call(tool_name, tool_args)
            step_info = {
                "iteration": iteration,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "status": "success",
                "result": tool_result,
            }
            executed_tools.append(step_info)
        except Exception as exc:
            step_info = {
                "iteration": iteration,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "status": "error",
                "error": str(exc),
            }
            executed_tools.append(step_info)
            scratchpad.append(f"Observation {iteration}: Tool failed - {str(exc)}")
            
            # Add error to conversation and continue
            messages.append({"role": "assistant", "content": f"ACTION: {tool_name} {json.dumps(tool_args)}"})
            messages.append({"role": "tool", "name": tool_name, "content": json.dumps({"error": str(exc)})})
            continue
        
        # Observe tool results
        observation = json.dumps(tool_result, default=str)
        scratchpad.append(f"Observation {iteration}: Tool executed successfully (output length: {len(observation)} chars).")
        
        # Add tool invocation and result to conversation history
        messages.append({"role": "assistant", "content": f"ACTION: {tool_name} {json.dumps(tool_args)}"})
        messages.append({"role": "tool", "name": tool_name, "content": observation})
        
        # Check if we should continue or stop
        if iteration >= max_iterations:
            scratchpad.append(f"Reached max iterations ({max_iterations}). Requesting final answer.")
            # Request final answer on last iteration
            final_response = TOOL_LLM.generate_with_tools(
                prompt="Summarize your analysis findings based on all tool executions. Provide a final answer.",
                tools=AVAILABLE_TOOLS,
                messages=messages,
            )
            final_answer = final_response.text if final_response.type == "text" else observation
            break
    
    # Compile final answer with execution details
    if final_answer is None:
        final_answer = "Analysis completed without final answer. Review executed steps above."
    
    answer_text = final_answer.strip()
    
    # Enhance answer with execution summary if tools were used
    if executed_tools:
        tool_summary = "\n\nExecution Summary:"
        for step in executed_tools:
            if step["status"] == "success":
                result = step.get("result", {})
                stdout = result.get("stdout", "").strip()[:200]
                tool_summary += f"\n- {step['tool_name']} (iteration {step['iteration']}): Success"
                if stdout:
                    tool_summary += f" | Output: {stdout}"
                if result.get("chart_created"):
                    tool_summary += " | Chart generated"
            else:
                tool_summary += f"\n- {step['tool_name']} (iteration {step['iteration']}): Error - {step.get('error', 'Unknown')}"
        answer_text += tool_summary
    
    # Extract code from last successful execution if available
    extracted_code = None
    for step in reversed(executed_tools):
        if step["status"] == "success" and step["tool_name"] == "python_execute":
            extracted_code = step["tool_args"].get("code")
            break
    
    # Save to memory
    MEMORY.add_turn(prompt, answer_text)
    
    return {
        "answer": answer_text,
        "analysis": answer_text,
        "python_code": extracted_code,
        "iterations": len([s for s in executed_tools if s["status"] == "success"]),
        "max_iterations": max_iterations,
        "scratchpad": scratchpad,
        "executed_tools": executed_tools,
        "execution_details": {
            "tool_count": len(executed_tools),
            "error_count": len([s for s in executed_tools if s["status"] == "error"]),
        },
    }


def _analyze_with_llm(
    prompt: str,
    db_path: Optional[str] = None,
    table_json: Optional[List[Dict[str, Any]]] = None,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Analyze using the LangChain-backed SQL analyst.

    Args:
        prompt: User question
        db_path: Optional SQLite database path
        table_json: Optional table data
        max_iterations: Max reasoning iterations (default 5)

    Returns:
        Analysis results with answer text
    """
    if db_path:
        answer = _run_db_file_langchain(db_path, prompt)
        return {"answer": answer}
    if table_json is not None:
        answer = _run_pandas_dataframe_agent(table_json, prompt, max_iterations=max_iterations)
        return {"answer": answer}
    return {"error": "No database file or table JSON provided."}


def analyze_table_json(table_json: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(table_json)
    preview = df.head(10).to_dict(orient="records")
    stats = df.describe(include="all").to_dict()
    schema = [{"column": c, "dtype": str(df[c].dtype)} for c in df.columns]
    return {"rows": len(df), "preview": preview, "stats": stats, "schema": schema}


def run_sql_analyst(
    db_file: Optional[str],
    sql_query: Optional[str],
    table_json: Optional[List[Dict[str, Any]]],
    prompt: str,
) -> Dict[str, Any]:
    if sql_query:
        if db_file:
            result = run_sql_query(db_file, sql_query)
            MEMORY.add_turn(prompt, json.dumps(result, default=str))
            return result
        if table_json is not None:
            result = run_sql_query_on_table(table_json, sql_query)
            MEMORY.add_turn(prompt, json.dumps(result, default=str))
            return result
        return {"error": "SQL query provided but no data source was supplied."}

    if db_file:
        if not os.path.exists(db_file):
            return {"error": f"Database file not found: {db_file}"}
        return agent_reasoning_loop(prompt=prompt, db_path=db_file, max_iterations=5)

    if table_json is not None:
        if not table_json:
            return {"error": "Table JSON is empty."}
        return agent_reasoning_loop(prompt=prompt, table_json=table_json, max_iterations=5)

    return {"error": "No database file, SQL query, or table JSON provided."}
