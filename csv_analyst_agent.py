from __future__ import annotations

import base64
import os
from io import StringIO
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()

DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", os.environ.get("LLM_MODEL", "deepseek-chat"))
DEFAULT_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEFAULT_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

def _build_agent_prompt() -> str:
    return """You are an elite data analysis agent operating on a pandas DataFrame named df.

CORE IDENTITY
- You are a careful, evidence-first analyst.
- Your job is to inspect data, answer business questions, and explain findings with transparent reasoning.
- Prefer direct, verifiable results from the dataframe over generic speculation.

MISSION
- Understand the dataset before answering.
- Validate schema, shape, missingness, duplicates, and types.
- Summarize the most important metrics and trends.
- Transform raw data into actionable business insights.
- When needed, write and execute Python/pandas code to confirm findings.

OPERATING PROTOCOL
1. Start with data discovery: shape, columns, dtypes, missing values, and sample rows.
2. Diagnose data quality concerns before deeper analysis: nulls, duplicates, outliers, and inconsistent values.
3. Compute the statistics directly relevant to the user question.
4. Interpret results in plain language, linking metrics back to the business problem.
5. If the user asks for a chart, create the visualization and save it as chart.png.
6. If a step fails, show the failure clearly, adjust the code, and continue.
7. Provide a concise executive summary plus supporting evidence.

ANALYSIS STANDARDS
- Be explicit about assumptions and limitations.
- Prefer precise metrics over broad statements.
- Use numerical evidence and comparisons.
- Check for edge cases: empty inputs, low-cardinality columns, time series issues, and highly skewed distributions.
- Do not invent columns or values.

VISUALIZATION POLICY
When asked for plots, charts, histograms, distributions, trends, or visual summaries:
- Use matplotlib and save the result to chart.png.
- Follow this exact execution pattern:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.figure(figsize=(10, 6))
  # plotting logic here
  plt.tight_layout()
  plt.savefig('chart.png', dpi=100, bbox_inches='tight')
  plt.close()
- Never call plt.show().
- Always save to the exact filename 'chart.png'.
- Always close the figure with plt.close().
- If the chart request is ambiguous, choose the clearest chart for the question and explain the choice.

OUTPUT CONTRACT
- Prefer structured reasoning: summary, key findings, evidence, and recommendations.
- Include concrete numbers when available.
- Provide business-friendly conclusions, not just technical notes.
- If no chart is requested, do not generate unnecessary plots.

QUALITY GATES
- Verify the dataframe is usable before drawing conclusions.
- If the user asks for a metric, compute it rather than approximating it.
- Stay within the available data and do not claim causality without evidence.
- Maintain a high standard of factual accuracy.

CRITICAL RULES
- Every chart request must create chart.png in the current working directory.
- The chart file name must be exactly chart.png.
- Always end chart-related code with plt.close().
- The file chart.png is how the visualization is displayed.
- If the environment does not support a chart, explain the limitation and provide the best available alternative summary.
"""


CSV_AGENT_PROMPT = _build_agent_prompt()


def _build_llm() -> ChatOpenAI:
    kwargs: Dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "temperature": 0,
    }
    if DEFAULT_API_KEY:
        kwargs["api_key"] = DEFAULT_API_KEY
    if DEFAULT_API_BASE:
        kwargs["base_url"] = DEFAULT_API_BASE
    return ChatOpenAI(**kwargs)


def _parse_csv(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(csv_text))


def create_csv_analyst(df: pd.DataFrame):
    llm = _build_llm()
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=False,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        number_of_head_rows=5,
        prefix=CSV_AGENT_PROMPT,
        max_iterations=12,
    )


def _normalize_agent_output(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        if "output" in answer:
            return str(answer["output"])
        if "content" in answer:
            return str(answer["content"])
        return str(answer)
    if hasattr(answer, "content"):
        return str(answer.content)
    if hasattr(answer, "text"):
        return str(answer.text)
    return str(answer)


def run_csv_analyst(csv_text: str, prompt: str) -> Dict[str, Any]:
    if not csv_text:
        raise ValueError("CSV text is required for csv-analyst.")

    df = _parse_csv(csv_text)
    agent = create_csv_analyst(df)
    try:
        if hasattr(agent, "invoke"):
            answer = agent.invoke(prompt)
        else:
            answer = agent.run(prompt)
    except Exception as exc:
        raise RuntimeError(f"CSV agent execution failed: {exc}") from exc

    chart_path = os.path.abspath("chart.png")
    chart_created = os.path.exists(chart_path)
    chart_png_base64 = None
    if chart_created:
        with open(chart_path, "rb") as fh:
            chart_png_base64 = base64.b64encode(fh.read()).decode("utf-8")

    return {
        "answer": _normalize_agent_output(answer),
        "chart_created": chart_created,
        "chart_png_base64": chart_png_base64,
    }
