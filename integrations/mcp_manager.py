from __future__ import annotations
import csv
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

logger = logging.getLogger(__name__)

from tools import (
    ClientOnboardingTool,
    EmailInboxTool,
    EmailSendTool,
    GoogleCalendarTool,
    GoogleDriveTool,
    GoogleSheetsTool,
    LoanReportDriveTool,
    PaymentReminderTool,
    PythonExecutionTool,
    RepaymentScheduleTool,
    SlackTool,
    TeamsTool,
    ToolBase,
    VisualizationTool,
    WordReportTool,
)


class MCPError(Exception):
    pass


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolBase] = {}
        self.register_default_tools()

    def register(self, tool: ToolBase) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> ToolBase:
        if name not in self.tools:
            raise MCPError(f'Tool not found: {name}')
        return self.tools[name]

    def list(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                'description': tool.description,
                'cooldown': tool.cooldown,
                'last_used_at': tool.last_used_at,
                'schema': getattr(tool, 'args_schema', {}),
                'configured': tool.is_configured(),
                'missing_credentials': tool.missing_credentials(),
                'credential_requirements': tool.credential_details(),
            }
            for name, tool in self.tools.items()
        }

    def to_openai_tools(self, names: List[str] | None = None) -> List[dict]:
        tools = self.tools.values() if names is None else [self.get(n) for n in names]
        return [t.to_openai_tool() for t in tools]

    def register_default_tools(self) -> None:
        self.register(EmailSendTool())
        self.register(EmailInboxTool())
        self.register(PaymentReminderTool())
        self.register(ClientOnboardingTool())
        self.register(PythonExecutionTool())
        self.register(RepaymentScheduleTool())
        self.register(VisualizationTool())
        self.register(WordReportTool())
        self.register(LoanReportDriveTool())
        self.register(GoogleDriveTool())
        self.register(GoogleSheetsTool())
        self.register(GoogleCalendarTool())
        self.register(SlackTool())
        self.register(TeamsTool())


class MCPManager:
    FALLBACK_TOOL_MAP = {
        'slack_post': ['email_send', 'teams_post'],
        'google_sheets_append': ['__csv_export__', 'word_report'],
    }
    MAX_RETRIES = 3

    def __init__(self, registry: Optional[ToolRegistry] = None, llm: Any = None):
        self.registry = registry or ToolRegistry()
        self._cooldown_queue: List[Dict[str, Any]] = []
        self._audit_log: List[Dict[str, Any]] = []
        self.llm = llm

    def call_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        attempted_tools: List[str] = []
        current_tool_name = tool_name
        current_kwargs = dict(kwargs)
        last_error: Optional[Exception] = None
        last_analysis = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            attempted_tools.append(current_tool_name)
            if current_tool_name != '__csv_export__':
                tool = self.registry.get(current_tool_name)

                if not tool.is_configured():
                    missing = tool.missing_credentials()
                    message = (
                        f'[{current_tool_name}] Tool not configured. Missing credentials: {", ".join(missing)}'
                    )
                    self._audit_log.append({
                        'tool': current_tool_name,
                        'args_summary': {k: str(v)[:80] for k, v in current_kwargs.items()},
                        'status': 'failed',
                        'error': message,
                        'attempt': attempt,
                        'timestamp': datetime.utcnow().isoformat(),
                    })
                    raise MCPError(message)

                if getattr(tool, 'args_schema', None):
                    try:
                        jsonschema.validate(
                            current_kwargs,
                            tool.args_schema,
                            format_checker=jsonschema.FormatChecker(),
                        )
                    except jsonschema.ValidationError as exc:
                        path = '.'.join(str(p) for p in exc.path) if exc.path else '(root)'
                        message = f'[{current_tool_name}] Invalid args — {path}: {exc.message}'
                        self._audit_log.append({
                            'tool': current_tool_name,
                            'args_summary': {k: str(v)[:80] for k, v in current_kwargs.items()},
                            'status': 'failed',
                            'error': message,
                            'attempt': attempt,
                            'timestamp': datetime.utcnow().isoformat(),
                        })
                        raise MCPError(message) from exc

                if not tool.can_run():
                    elapsed = time.time() - tool.last_used_at if tool.last_used_at else 0.0
                    retry_in = max(0.0, tool.cooldown - elapsed)
                    entry = {
                        'tool_name': current_tool_name,
                        'kwargs': current_kwargs,
                        'queued_at': time.time(),
                        'retry_in': retry_in,
                    }
                    self._cooldown_queue.append(entry)
                    self._audit_log.append({
                        'tool': current_tool_name,
                        'args_summary': {k: str(v)[:80] for k, v in current_kwargs.items()},
                        'status': 'queued',
                        'retry_in_seconds': round(retry_in, 1),
                        'attempt': attempt,
                        'timestamp': datetime.utcnow().isoformat(),
                    })
                    return {
                        'status': 'queued',
                        'tool': current_tool_name,
                        'retry_in_seconds': round(retry_in, 1),
                        'queued': True,
                    }

            start = time.time()
            try:
                if current_tool_name == '__csv_export__':
                    result = self._execute_internal_csv_export(current_kwargs)
                else:
                    result = tool.execute(**current_kwargs)

                if isinstance(result, dict) and result.get('error'):
                    raise ToolError(result.get('error'))

                duration_ms = int((time.time() - start) * 1000)
                if current_tool_name != '__csv_export__':
                    tool.update_last_used()
                self._audit_log.append({
                    'tool': current_tool_name,
                    'args_summary': {k: str(v)[:80] for k, v in current_kwargs.items()},
                    'status': 'success',
                    'duration_ms': duration_ms,
                    'attempt': attempt,
                    'timestamp': datetime.utcnow().isoformat(),
                })

                if isinstance(result, dict) and (attempt > 1 or attempted_tools != [tool_name]):
                    result['_recovery'] = {
                        'attempts': attempt,
                        'tool_path': attempted_tools,
                        'analysis': last_analysis,
                    }
                return result
            except Exception as exc:
                last_error = exc
                duration_ms = int((time.time() - start) * 1000)
                last_analysis = self._analyze_failure(current_tool_name, current_kwargs, exc, attempt)
                self._audit_log.append({
                    'tool': current_tool_name,
                    'args_summary': {k: str(v)[:80] for k, v in current_kwargs.items()},
                    'status': 'failed',
                    'error': str(exc),
                    'analysis': last_analysis,
                    'duration_ms': duration_ms,
                    'attempt': attempt,
                    'timestamp': datetime.utcnow().isoformat(),
                })

                if attempt >= self.MAX_RETRIES:
                    break

                corrected_kwargs = self._suggest_retry_args(
                    current_tool_name,
                    current_kwargs,
                    exc,
                    last_analysis,
                )
                if corrected_kwargs != current_kwargs:
                    self._audit_log.append({
                        'tool': current_tool_name,
                        'status': 'retry',
                        'reason': last_analysis,
                        'attempt': attempt + 1,
                        'args_summary': {k: str(v)[:80] for k, v in corrected_kwargs.items()},
                        'timestamp': datetime.utcnow().isoformat(),
                    })
                    current_kwargs = corrected_kwargs
                    continue

                fallback_tool = self._get_alternative_tool(current_tool_name, attempted_tools)
                if fallback_tool:
                    self._audit_log.append({
                        'tool': current_tool_name,
                        'status': 'fallback',
                        'reason': last_analysis,
                        'fallback_to': fallback_tool,
                        'attempt': attempt + 1,
                        'timestamp': datetime.utcnow().isoformat(),
                    })
                    current_tool_name = fallback_tool
                    current_kwargs = self._remap_args_for_alternative(fallback_tool, current_kwargs)
                    continue

                break

        error_type = type(last_error).__name__ if last_error is not None else 'Error'
        raise MCPError(
            f'[{tool_name}] Failed after {len(attempted_tools)} attempts. '
            f'Last error: {error_type}: {str(last_error)}. Analysis: {last_analysis}'
        )

    def _analyze_failure(
        self,
        tool_name: str,
        args: Dict[str, Any],
        exc: Exception,
        attempt: int,
    ) -> str:
        message = str(exc)
        if self.llm:
            prompt = (
                f"Tool '{tool_name}' failed on attempt {attempt}.\n"
                f"Args: {args}\n"
                f"Error: {message}\n\n"
                "Briefly describe why this failed and whether parameters, credentials, or a transient issue caused it."
            )
            try:
                analysis = self.llm.generate(prompt, max_tokens=120, temperature=0.2)
                return analysis.strip()
            except Exception:
                pass

        lowered = message.lower()
        if 'missing credentials' in lowered or 'not configured' in lowered:
            return 'Missing or invalid tool credentials.'
        if 'timeout' in lowered or 'network' in lowered or 'unreachable' in lowered:
            return 'Transient network or service issue occurred.'
        if 'invalid' in lowered or 'required' in lowered or 'schema' in lowered:
            return 'Tool arguments were invalid or missing required fields.'
        return 'Unknown failure; the tool raised an exception.'

    def _suggest_retry_args(
        self,
        tool_name: str,
        args: Dict[str, Any],
        exc: Exception,
        analysis: str,
    ) -> Dict[str, Any]:
        if 'credentials' in analysis.lower() or 'not configured' in analysis.lower():
            return args
        if self.llm:
            prompt = (
                f"You are an assistant helping to repair a failed tool call.\n"
                f"Tool: {tool_name}\n"
                f"Current args: {args}\n"
                f"Failure analysis: {analysis}\n"
                "If a small parameter correction can fix this, return only a JSON object of corrected args. "
                "If no correction is possible, return the original args exactly."
            )
            try:
                response = self.llm.generate(prompt, max_tokens=150, temperature=0.2)
                response = response.strip()
                if not response.startswith('{'):
                    # attempt to extract JSON from text
                    import re
                    match = re.search(r'\{.*\}', response, re.S)
                    if match:
                        response = match.group(0)
                corrected = json.loads(response)
            except Exception:
                corrected = None
            if isinstance(corrected, dict):
                return corrected
        return args

    def _get_alternative_tool(
        self,
        failed_tool: str,
        attempted_tools: List[str],
    ) -> Optional[str]:
        for candidate in self.FALLBACK_TOOL_MAP.get(failed_tool, []):
            if candidate not in attempted_tools:
                return candidate
        return None

    def _remap_args_for_alternative(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        if tool_name == 'email_send':
            return {
                'to': args.get('to') or args.get('email') or args.get('recipient') or '',
                'subject': args.get('subject') or 'Fallback message from microfinance agent',
                'body': args.get('message') or args.get('body') or str(args),
            }
        if tool_name == 'teams_post':
            return {
                'message': args.get('message') or args.get('body') or str(args),
                'channel': args.get('channel') or args.get('room') or '',
            }
        if tool_name == '__csv_export__':
            return {
                'rows': args.get('values') or args.get('rows') or [],
                'output_path': args.get('output_path') or 'fallback_export.csv',
            }
        if tool_name == 'word_report':
            values = args.get('values') or []
            rows = []
            if isinstance(values, list):
                for row in values:
                    if isinstance(row, (list, tuple)):
                        rows.append(', '.join(str(cell) for cell in row))
                    else:
                        rows.append(str(row))
            else:
                rows.append(str(values))
            return {
                'title': 'Fallback export from Google Sheets',
                'sections': [
                    {
                        'heading': 'Sheet export',
                        'content': '\n'.join(rows) or 'No data available.',
                    }
                ],
            }
        return args

    def _execute_internal_csv_export(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        rows = kwargs.get('rows')
        if not isinstance(rows, list) or not rows:
            raise ToolError('CSV export requires a non-empty list of rows.')
        output_path = kwargs.get('output_path')
        if not output_path:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            output_path = temp_file.name
            temp_file.close()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            for row in rows:
                if isinstance(row, (list, tuple)):
                    writer.writerow([str(item) for item in row])
                else:
                    writer.writerow([str(row)])
        return {'status': 'csv_exported', 'output_path': str(output_file), 'rows_exported': len(rows)}

    def flush_queue(self) -> List[Dict[str, Any]]:
        """Execute queued tool calls whose cooldown has expired. Call at start of each planner step."""
        MAX_AGE_SECONDS = 120
        now = time.time()
        still_pending: List[Dict[str, Any]] = []
        flushed: List[Dict[str, Any]] = []

        for entry in self._cooldown_queue:
            age = now - entry['queued_at']
            if age > MAX_AGE_SECONDS:
                self._audit_log.append({
                    'tool': entry['tool_name'],
                    'status': 'queue_expired',
                    'age_seconds': round(age, 1),
                    'timestamp': datetime.utcnow().isoformat(),
                })
                continue

            try:
                tool = self.registry.get(entry['tool_name'])
            except MCPError as exc:
                self._audit_log.append({
                    'tool': entry['tool_name'],
                    'status': 'missing_on_flush',
                    'error': str(exc),
                    'timestamp': datetime.utcnow().isoformat(),
                })
                continue

            if tool.can_run():
                try:
                    result = self.call_tool(entry['tool_name'], **entry['kwargs'])
                    flushed.append({
                        'tool': entry['tool_name'],
                        'result': result,
                        'queued_at': entry['queued_at'],
                    })
                except MCPError as exc:
                    self._audit_log.append({
                        'tool': entry['tool_name'],
                        'status': 'flush_failed',
                        'error': str(exc),
                        'timestamp': datetime.utcnow().isoformat(),
                    })
            else:
                still_pending.append(entry)

        self._cooldown_queue = still_pending
        return flushed

    def available_tools(self) -> Dict[str, Any]:
        return self.registry.list()

    def get_tool_schemas(self, names: List[str] | None = None) -> List[dict]:
        return self.registry.to_openai_tools(names)

    def route_agent(self, agent_type: str, action: str, **kwargs: Any) -> Dict[str, Any]:
        """Deprecated: use MCPManager.call_tool(tool_name, **kwargs) directly."""
        raise MCPError(
            "route_agent() is deprecated. Call MCPManager.call_tool() with the actual tool name "
            "instead of using agent_type/action routing."
        )
