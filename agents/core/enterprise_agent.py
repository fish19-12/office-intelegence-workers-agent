from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from chunker import chunk_text
from embeddings_rag import AdvancedRAG
from ingestion import ingest_file
from supabase_client import SupabaseClient
try:
    from langdetect import detect
except Exception:
    def detect(text: str) -> str:  # type: ignore
        return 'unknown'
from llm_interface import LLMFactory
from mcp_manager import MCPError, MCPManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class DocumentAsset:
    id: str
    path: Path
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Dict[str, Any]] = field(default_factory=list)


class EnterpriseFileAgent:
    DEFAULT_TASK_CATEGORIES = [
        'Reporting',
        'Customer Handling',
        'Risk Assessment',
        'Collections',
        'Compliance',
        'Operations',
    ]

    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        rag_model: str = 'BAAI/bge-m3',
        upload_dir: str = 'uploads',
        supabase_table: str = 'documents',
        enable_supabase: bool = True,
    ):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.llm = LLMFactory.create('deepseek', llm_config or {})
        self.last_llm_success_timestamp: Optional[str] = None
        # Allow overriding embedding model via EMBEDDING_MODEL env var
        rag_model = os.environ.get('EMBEDDING_MODEL') or rag_model
        # Use AdvancedRAG and provide LLM for decomposition/compression
        self.rag = AdvancedRAG(llm=self.llm, embedding_model_name=rag_model)
        self.session_warning: Optional[str] = None
        self.mcp = MCPManager(llm=self.llm)
        self.documents: Dict[str, DocumentAsset] = {}
        self.memory: List[Dict[str, Any]] = []
        self.memory_summary: str = ''
        self.conversation_history: List[Dict[str, Any]] = []
        self.task_categories = set(self.DEFAULT_TASK_CATEGORIES)
        self.supabase_store = None
        if enable_supabase:
            try:
                self.supabase_store = SupabaseClient(table_name=supabase_table)
                # fetch remote count and set rag remote count for sync tracking
                try:
                    remote_count = self.supabase_store.count_documents()
                    self.rag._remote_count = int(remote_count or 0)
                except Exception:
                    self.rag._remote_count = 0
            except Exception as exc:
                logger.warning('Supabase disabled because it could not be configured: %s', exc)
                self.supabase_store = None

    def _make_document_asset(self, path: Path, metadata: Optional[Dict[str, Any]] = None) -> DocumentAsset:
        file_text = ingest_file(path)
        document_id = str(uuid.uuid4())
        full_metadata = {
            'source_path': str(path),
            'filename': path.name,
            **(metadata or {}),
        }
        chunks = []
        for idx, chunk in enumerate(chunk_text(file_text, chunk_size=1600, overlap=200)):
            chunk_metadata = {
                'source_path': str(path),
                'chunk_index': idx,
                'filename': path.name,
                **full_metadata,
            }
            chunks.append({
                'id': f'{document_id}-{idx}',
                'text': chunk,
                'metadata': chunk_metadata,
            })
        return DocumentAsset(id=document_id, path=path, text=file_text, metadata=full_metadata, chunks=chunks)

    def ingest_file(
        self,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
        task_type: Optional[str] = None,
        user_goal: Optional[str] = None,
    ) -> DocumentAsset:
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f'File not found: {path}')
        metadata = metadata.copy() if metadata else {}
        # Detect language of document and attach to metadata for multilingual support
        try:
            file_text_for_lang = ingest_file(source_path)
            detected = detect(file_text_for_lang)
            if detected and detected != 'en':
                metadata['language'] = detected
        except Exception:
            pass
        if category:
            metadata['category'] = category
            self.task_categories.add(category)
        if task_type:
            metadata['task_type'] = task_type
            self.task_categories.add(task_type)
        if user_goal:
            metadata['user_goal'] = user_goal
        document = self._make_document_asset(source_path, metadata=metadata)
        self.documents[document.id] = document
        self._index_document(document)
        self.add_memory_event('document_ingested', {
            'document_id': document.id,
            'filename': source_path.name,
            'category': category or task_type,
            'user_goal': user_goal,
        })
        return document

    def _index_document(self, document: DocumentAsset) -> None:
        tuples = [(chunk['text'], chunk['metadata']) for chunk in document.chunks]
        # Incrementally add new chunks to the local RAG index
        self.rag.add_documents(tuples)
        if self.supabase_store is not None:
            try:
                self.supabase_store.upsert_documents([
                    {
                        'id': chunk['id'],
                        'text': chunk['text'],
                        'metadata': chunk['metadata'],
                        'embedding': self.rag.embeddings.embed_documents([chunk['text']])[0],
                    }
                    for chunk in document.chunks
                ])
            except Exception as exc:
                # mark a session warning and log specific guidance
                msg = f'Supabase sync failed for {document.id} — local index is ahead of remote. Re-upload to sync.'
                self.session_warning = msg
                logger.warning(msg + ' Details: %s', exc)

    def _merge_search_results(self, local_results: List[Dict[str, Any]], external_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        merged_by_id = {}
        for item in local_results + external_results:
            item_id = item['meta'].get('id') or item['meta'].get('chunk_id') or f"{item['meta'].get('filename')}-{len(merged_by_id)}"
            existing = merged_by_id.get(item_id)
            if existing is None or item['score'] > existing['score']:
                merged_by_id[item_id] = item
        merged = sorted(merged_by_id.values(), key=lambda x: x['score'], reverse=True)
        return merged[:top_k]

    def retrieve_context(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        # Use AdvancedRAG.retrieve() when available, otherwise fall back to legacy query()
        try:
            if hasattr(self.rag, "retrieve"):
                res = self.rag.retrieve(query, top_k=top_k)
                local_results = res.get("items", []) if isinstance(res, dict) else res
            else:
                local_results = self.rag.query(query, topk=top_k, metadata_filter=metadata_filter)
        except Exception:
            local_results = []

        if self.supabase_store is None:
            return local_results
        try:
            supabase_results = self.supabase_store.query_documents(query, self.rag.embeddings, top_k=top_k, metadata_filter=metadata_filter)
            if not supabase_results:
                return local_results
            return self._merge_search_results(local_results, supabase_results, top_k)
        except Exception as exc:
            logger.warning('Supabase retrieval failed: %s', exc)
            return local_results

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve_context(query, top_k=top_k)

    def _get_memory_context_prompt(self) -> str:
        recent_activity = self.get_memory_summary(30)
        long_term = ''
        if self.memory_summary:
            long_term = f'[Long-term context]\n{self.memory_summary}\n\n'
        return f"{long_term}[Recent activity]\n{recent_activity}"

    def _build_prompt(self, query: str, top_context: List[Dict[str, Any]], user_goal: Optional[str] = None) -> str:
        context = '\n\n'.join([f"[{item['meta'].get('filename', 'unknown')}] {item['text']}" for item in top_context])
        memory_context = self._get_memory_context_prompt()
        return (
            'You are MFIs, a microfinance business agent that stores documents, remembers past interactions, and plans actions with tools. '
            'Use the memory history, document category metadata, and the provided context to answer the user request. Do not invent unsupported details.\n\n'
            f'Agent goal: {user_goal or "General microfinance assistance"}\n\n'
            f'{memory_context}\n\n'
            f'Context:\n{context}\n\n'
            f'User request:\n{query}\n\n'
            'Provide a concise, structured response and, when asked for a report, include python code in a ```python``` block for analysis or visualization.'
        )

    def _flush_memory_to_long_term(self) -> None:
        if not self.memory:
            return
        log_lines = [f"{item['timestamp']} - {item['event']}: {item['detail']}" for item in self.memory]
        prompt = (
            'Summarize the following agent activity log into 3-5 key facts that would be useful to remember for future tasks. '
            'Be specific about document names, customer IDs, amounts, and actions taken. Output only the summary, no preamble.\n\n'
            + '\n'.join(log_lines)
        )
        try:
            summary = self.llm.generate(prompt, max_tokens=300)
            self.memory_summary = summary.strip()
            if self.supabase_store is not None:
                session_id = os.environ.get('AGENT_SESSION_ID', 'default')
                try:
                    self.supabase_store.upsert_agent_session(session_id, self.memory_summary)
                except Exception as exc:
                    logger.warning('Failed to persist agent session summary: %s', exc)
        except Exception as exc:
            logger.warning('Failed to summarize memory log: %s', exc)
            self.memory_summary = self.memory_summary or ''
        self.memory = []

    def _record_llm_response(self, response: str) -> None:
        if not response:
            return
        lower_text = response.lower()
        failure_markers = ['error', 'request error', 'unable to', 'could not', 'exception']
        if any(marker in lower_text for marker in failure_markers):
            return
        self.last_llm_success_timestamp = datetime.utcnow().isoformat() + 'Z'

    def answer_query(self, query: str, user_goal: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        result = self.reactive_run(query, user_goal=user_goal, top_k=top_k)
        self.add_memory_event('query', {
            'query': query,
            'user_goal': user_goal,
            'answer': result.get('answer'),
            'tool_steps': result.get('tool_steps', []),
        })
        self.record_conversation_entry('user', query, {'goal': user_goal, 'type': 'semantic_query'})
        self.record_conversation_entry('assistant', result.get('answer', ''), {'tool_steps': result.get('tool_steps', [])})
        return result

    def _available_tools_text(self) -> str:
        tools = self.mcp.available_tools()
        formatted_tools = []
        for name, info in tools.items():
            schema_text = json.dumps(info.get('schema', {}), indent=2)
            formatted_tools.append(
                f"{name}: {info['description']}\nArgs schema:\n{schema_text}"
            )
        return '\n\n'.join(formatted_tools)

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.mcp.call_tool(tool_name, **args)

    def _get_env_max_steps(self) -> int:
        raw_value = os.environ.get('AGENT_MAX_STEPS')
        try:
            if raw_value is None:
                return 10
            parsed = int(raw_value)
        except (TypeError, ValueError):
            return 10
        return min(max(parsed, 1), 20)

    def _invoke_llm(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        result = self.llm.generate(prompt, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            text, metadata = result
        elif isinstance(result, dict):
            text = result.get('text') or result.get('output') or result.get('response') or result.get('result') or str(result)
            metadata = result.get('usage', {}) if isinstance(result.get('usage', {}), dict) else {}
        else:
            text = str(result)
            metadata = {}
        return text, metadata

    def _summarize_action(self, response: str, action_name: Optional[str]) -> str:
        if action_name:
            return f'Requested tool action {action_name}.'
        concise = response.strip().splitlines()[0] if response and response.strip() else ''
        sentence = re.split(r'(?<=[.!?])\s+', concise.strip())[0]
        return sentence or 'No action summary available.'

    def _guess_agent_type(self, text: str) -> str:
        text = (text or '').lower()
        if any(keyword in text for keyword in ['loan', 'payment', 'customer', 'reminder', 'collection']):
            return 'loan_officer'
        if any(keyword in text for keyword in ['compliance', 'email', 'calendar', 'meeting', 'audit']):
            return 'compliance'
        if any(keyword in text for keyword in ['drive', 'sheet', 'slack', 'teams', 'upload', 'share']):
            return 'operations'
        if any(keyword in text for keyword in ['analysis', 'visualization', 'chart', 'summary', 'report']):
            return 'analytics'
        return 'operations'

    def _build_reactive_prompt(
        self,
        query: str,
        top_context: List[Dict[str, Any]],
        user_goal: Optional[str] = None,
        observations: Optional[List[str]] = None,
    ) -> str:
        context = '\n\n'.join([f"[{item['meta'].get('filename', 'unknown')}] {item['text']}" for item in top_context])
        memory_context = self._get_memory_context_prompt()
        observations_text = '\n'.join(observations) if observations else 'None'
        return (
            'You are MFIs, a reactive microfinance agent. You must respond using available tools when required and stop with a final answer when your task is complete. '
            'Do not invent unsupported details.\n\n'
            'Available tools:\n'
            f'{self._available_tools_text()}\n\n'
            'When you decide to use a tool, reply with a single line in this exact format:\n'
            '`ACTION: <tool_name> <json_encoded_args>`\n'
            'When you are finished, reply with a single line in this exact format:\n'
            '`FINAL: <answer>`\n\n'
            f'Agent goal: {user_goal or "General microfinance assistance"}\n\n'
            f'{memory_context}\n\n'
            f'Context:\n{context}\n\n'
            f'Observations:\n{observations_text}\n\n'
            f'User request:\n{query}\n\n'
        )

    def _parse_reactive_response(self, response: str) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
        action_name = None
        action_args: Dict[str, Any] = {}
        final_answer = None
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith('ACTION:'):
                payload = line.split(':', 1)[1].strip()
                if not payload:
                    continue
                parts = payload.split(None, 1)
                action_name = parts[0].strip()
                if len(parts) > 1:
                    try:
                        action_args = json.loads(parts[1].strip())
                    except json.JSONDecodeError:
                        action_args = {'args': parts[1].strip()}
            elif line.upper().startswith('FINAL:'):
                final_answer = line.split(':', 1)[1].strip()
        return action_name, action_args, final_answer

    def reactive_run(
        self,
        query: str,
        user_goal: Optional[str] = None,
        top_k: int = 5,
        max_steps: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
    ) -> Dict[str, Any]:
        if max_steps is None:
            max_steps = self._get_env_max_steps()
        else:
            max_steps = min(max(max_steps, 1), 20)

        search_results = self.retrieve_context(query, top_k=top_k)
        observations: List[str] = []
        tool_steps: List[Dict[str, Any]] = []
        steps_log: List[Dict[str, Any]] = []
        final_answer: Optional[str] = None
        response_text: str = ''

        for step in range(1, max_steps + 1):
            flushed = self.mcp.flush_queue()
            if flushed:
                observations.append(f'Flushed {len(flushed)} queued tool call(s) whose cooldown expired.')

            prompt = self._build_reactive_prompt(query, search_results, user_goal=user_goal, observations=observations)
            step_start = time.perf_counter()
            response_text, response_metadata = self._invoke_llm(prompt, max_tokens=900)
            self._record_llm_response(response_text)
            self.add_memory_event('reactive_step', {'step': step, 'response': response_text})
            action_name, action_args, final_answer = self._parse_reactive_response(response_text)
            action_summary = self._summarize_action(response_text, action_name)
            tokens_used = None
            if isinstance(response_metadata, dict):
                tokens_used = response_metadata.get('total_tokens') or response_metadata.get('usage', {}).get('total_tokens')
                if isinstance(tokens_used, str) and tokens_used.isdigit():
                    tokens_used = int(tokens_used)
            step_log = {
                'step': step,
                'tool': action_name,
                'action_summary': action_summary,
                'status': 'skipped',
                'tokens_used': int(tokens_used) if isinstance(tokens_used, int) else None,
                'duration_ms': 0,
            }

            if action_name:
                try:
                    tool_result = self._execute_tool(action_name, action_args)
                    if isinstance(tool_result, dict) and tool_result.get('status') == 'cooldown':
                        retry_in = tool_result.get('retry_in_seconds', 0.0)
                        observation = (
                            f"Tool {action_name} is cooling down. It will be available in {retry_in}s. "
                            'I will proceed with other steps and retry automatically.'
                        )
                        observations.append(observation)
                        tool_steps.append({
                            'action': action_name,
                            'args': action_args,
                            'result': tool_result,
                        })
                        step_log['status'] = 'skipped'
                        step_log['duration_ms'] = int((time.perf_counter() - step_start) * 1000)
                        steps_log.append(step_log)
                        if progress_callback:
                            progress_callback(step_log, step, max_steps)
                        continue
                    status = 'success'
                except MCPError as exc:
                    message = str(exc)
                    if 'args invalid:' in message:
                        invalid_detail = message.split('args invalid:', 1)[1].strip()
                        observation = f'Tool args invalid: {invalid_detail}'
                        observations.append(observation)
                        tool_result = {'error': observation}
                    else:
                        tool_result = {'error': message}
                    status = 'error'
                except Exception as exc:
                    tool_result = {'error': str(exc)}
                    status = 'error'
                observations.append(f'Tool {action_name} result: {json.dumps(tool_result, default=str)}')
                tool_steps.append({
                    'action': action_name,
                    'args': action_args,
                    'result': tool_result,
                })
                step_log['status'] = status
                step_log['duration_ms'] = int((time.perf_counter() - step_start) * 1000)
                steps_log.append(step_log)
                if progress_callback:
                    progress_callback(step_log, step, max_steps)
                continue

            step_log['status'] = 'success' if final_answer else 'skipped'
            step_log['duration_ms'] = int((time.perf_counter() - step_start) * 1000)
            steps_log.append(step_log)
            if progress_callback:
                progress_callback(step_log, step, max_steps)

            if final_answer:
                return {
                    'query': query,
                    'goal': user_goal,
                    'results': search_results,
                    'answer': final_answer,
                    'partial': False,
                    'completed_steps': step,
                    'steps_log': steps_log,
                    'llm_response': response_text,
                    'tool_steps': tool_steps,
                    'observations': observations,
                }
            return {
                'query': query,
                'goal': user_goal,
                'results': search_results,
                'answer': response_text,
                'partial': False,
                'completed_steps': step,
                'steps_log': steps_log,
                'llm_response': response_text,
                'tool_steps': tool_steps,
                'observations': observations,
            }

        summary_prompt = 'You have reached the step limit. Summarize what has been accomplished so far and what steps remain incomplete.'
        summary_text, _ = self._invoke_llm(summary_prompt, max_tokens=300)
        return {
            'query': query,
            'goal': user_goal,
            'results': search_results,
            'answer': summary_text,
            'partial': True,
            'completed_steps': max_steps,
            'steps_log': steps_log,
            'llm_response': response_text,
            'tool_steps': tool_steps,
            'observations': observations,
        }

    def get_memory_summary(self, max_events: int = 5) -> str:
        if not self.memory:
            return 'No recent activity stored yet.'
        recent = self.memory[-max_events:]
        return '\n'.join([f"{item['timestamp']} - {item['event']}: {item['detail']}" for item in recent])

    def get_full_memory_context(self) -> str:
        sections = []
        if self.memory_summary:
            sections.append(f'[Long-term context]\n{self.memory_summary}')
        sections.append(f'[Recent activity]\n{self.get_memory_summary(30)}')
        return '\n\n'.join(sections)

    def available_categories(self) -> List[str]:
        return sorted(self.task_categories)

    def add_memory_event(self, event: str, detail: Any) -> None:
        self.memory.append({
            'event': event,
            'detail': detail,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        })
        if len(self.memory) >= 30:
            self._flush_memory_to_long_term()

    def record_conversation_entry(self, role: str, text: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.conversation_history.append({
            'role': role,
            'text': text,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        })

    def get_conversation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.conversation_history[-limit:]

    def _select_tool_for_goal(self, user_goal: str, query: str) -> Tuple[Optional[str], Dict[str, Any]]:
        text = ' '.join([user_goal or '', query or '']).lower()
        if 'report' in text or 'loan performance' in text:
            return 'word_report', {
                'title': f'MFIs report',
                'sections': [
                    {'heading': 'Executive Summary', 'content': user_goal or query or 'Generate a microfinance report based on uploaded documents.'},
                ],
                'images': [],
                'output_path': f'mfis_report_{uuid.uuid4().hex}.docx',
            }
        if 'analysis' in text or 'visualization' in text or 'chart' in text:
            return 'python_execute', {
                'code': 'print("MFIs analysis task executed.")',
                'timeout': 60,
            }
        return None, {}

    def plan_and_execute(
        self,
        user_goal: str,
        query: Optional[str] = None,
        top_k: int = 5,
        max_steps: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
        agent_type_override: Optional[str] = None,
        confirm: Optional[bool] = None,
        confirmed_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.add_memory_event('goal_received', {
            'goal': user_goal,
            'query': query,
        })
        self.record_conversation_entry('user', user_goal, {'goal': user_goal, 'query': query, 'type': 'planner'})
        # Infer agent type (can be overridden by caller / UI)
        agent_type = agent_type_override or self._guess_agent_type(user_goal)

        # Dry-run: ask the LLM which irreversible tools it would use for this goal
        irreversible_tools = [
            'EmailSendTool',
            'PaymentReminderTool',
            'GoogleDriveTool',
            'GoogleSheetsTool',
            'ClientOnboardingTool',
            'SlackTool',
            'TeamsTool',
        ]
        dry_prompt = (
            f"Given this goal: {user_goal}\n"
            f"Which tools from this list would you use? Reply with only a JSON array of tool names.\n"
            f"List: {irreversible_tools}"
        )
        planned_tools: List[str] = []
        try:
            dry_resp = self.llm.generate(dry_prompt, max_tokens=200)
            try:
                planned_tools = json.loads(dry_resp)
                if not isinstance(planned_tools, list):
                    planned_tools = []
            except Exception:
                # Fallback: try to extract a JSON array substring
                m = re.search(r"\[.*?\]", str(dry_resp), flags=re.S)
                if m:
                    try:
                        planned_tools = json.loads(m.group(0))
                    except Exception:
                        planned_tools = []
        except Exception:
            planned_tools = []

        # Detect irreversible tool usage
        irreversible_in_plan = [t for t in planned_tools if t in irreversible_tools]

        # If irreversible tools are present and caller/UI hasn't confirmed, return a confirmation payload
        if irreversible_in_plan and not confirm:
            return {
                'requires_confirmation': True,
                'agent_type': agent_type,
                'goal_summary': user_goal,
                'tools_planned': planned_tools,
            }

        # Determine who confirmed this workflow start for logging
        if irreversible_in_plan:
            confirmer = confirmed_by or ('user' if confirm else 'auto')
        else:
            confirmer = confirmed_by or 'auto'

        # Log confirmed workflow start
        self.add_memory_event('workflow_confirmed', {
            'agent_type': agent_type,
            'goal_summary': user_goal,
            'tools_planned': planned_tools,
            'confirmed_by': confirmer,
        })

        # Proceed to reactive execution
        result = self.reactive_run(
            query or user_goal or '',
            user_goal=user_goal,
            top_k=top_k,
            max_steps=max_steps,
            progress_callback=progress_callback,
        )
        self.add_memory_event('planning', {'reactive_result': result})
        self.record_conversation_entry('assistant', result.get('answer', ''), {'tool_steps': result.get('tool_steps', []), 'type': 'planner'})
        return result

    def _extract_python_code_blocks(self, text: str) -> List[str]:
        return [match.strip() for match in re.findall(r'```python\s*(.*?)```', text, flags=re.DOTALL | re.IGNORECASE)]

    def _normalize_content(self, text: str) -> str:
        return text.strip().replace('\r\n', '\n')

    def create_advanced_report(
        self,
        file_path: str,
        user_request: str,
        output_path: Optional[str] = None,
        top_k: int = 5,
        run_without_review: bool = False,
    ) -> Dict[str, Any]:
        document = self.ingest_file(file_path)
        search_results = self.semantic_search(user_request, top_k=top_k)
        prompt = (
            f'You are a senior enterprise microfinance analyst. The user has uploaded {document.path.name}.'
            f' Their request: {user_request}\n\n'
            'Review the relevant document content and generate a detailed report plan. '
            'Return a python code block only inside ```python``` that performs data analysis or visualization on the uploaded file. '
            'Also provide an executive summary and suggested report sections. Do not include any code outside the python block.'
            '\n\nContext:\n'
            + '\n\n'.join([f"Section {i+1}: {item['text']}" for i, item in enumerate(search_results)])
        )
        agent_output = self.llm.generate(prompt, max_tokens=1200)
        self._record_llm_response(agent_output)

        code_blocks = self._extract_python_code_blocks(agent_output)
        summary_text = self._normalize_content(re.sub(r'```python\s*.*?```', '', agent_output, flags=re.DOTALL | re.IGNORECASE))

        pending_code_blocks = []
        for idx, block in enumerate(code_blocks):
            pending_code_blocks.append({
                'id': f'codeblock-{idx}-{uuid.uuid4().hex[:8]}',
                'section_name': f'Analysis {idx + 1}',
                'original_code': block,
                'editable_code': block,
                'execution_result': None,
                'include_in_report': True,
            })

        code_output = None
        generated_images: List[str] = []
        report_path = None
        if run_without_review and code_blocks:
            workspace = tempfile.mkdtemp(prefix='enterprise_report_')
            workspace_path = Path(workspace)
            workspace_path.mkdir(parents=True, exist_ok=True)
            generated_images = []
            code_output = []
            for block in code_blocks:
                tool_result = self.mcp.call_tool('python_execute', code=block, work_dir=str(workspace_path), timeout=120)
                code_output.append(tool_result)
            generated_images = [str(p) for p in workspace_path.glob('*.png')] + [str(p) for p in workspace_path.glob('*.jpg')]
            sections = [
                {'heading': 'Executive Summary', 'content': self._normalize_content(summary_text or 'Generated analysis for the uploaded document.')},
                {'heading': 'Key Findings', 'content': 'Use the analysis code output and generated charts to summarize the most important points.'},
            ]
            report_file = output_path or str(Path(workspace_path) / f'{document.path.stem}_report.docx')
            report_result = self.mcp.call_tool(
                'word_report',
                title=f'Advanced Report for {document.path.name}',
                sections=sections,
                images=generated_images,
                output_path=report_file,
            )
            report_path = report_result['output_path']

        return {
            'document_id': document.id,
            'file_path': str(document.path),
            'report_path': report_path,
            'generated_images': generated_images,
            'code_output': code_output,
            'analysis_summary': summary_text,
            'raw_agent_output': agent_output,
            'pending_code_blocks': pending_code_blocks,
        }

    def materialize_entire_document(self, query: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        output = self.answer_query(query)
        report_file = output_path or f'{uuid.uuid4().hex}_response.txt'
        Path(report_file).write_text(output['answer'], encoding='utf-8')
        return {'file_path': report_file, 'answer': output['answer']}
