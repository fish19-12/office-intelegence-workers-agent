import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from supabase import create_client
except ImportError as exc:
    raise ImportError('supabase package is required for SupabaseClient') from exc

logger = logging.getLogger(__name__)


class SupabaseClient:
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None, table_name: str = 'documents'):
        self.url = url or os.environ.get('SUPABASE_URL')
        self.key = key or os.environ.get('SUPABASE_KEY')
        self.table_name = table_name
        if not self.url or not self.key:
            raise RuntimeError('SUPABASE_URL and SUPABASE_KEY must be configured')
        self.client = create_client(self.url, self.key)

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> None:
        rows = []
        for doc in documents:
            rows.append({
                'id': doc['id'],
                'content': doc['text'],
                'metadata': doc.get('metadata', {}),
                'embedding': doc['embedding'],
            })
        self.client.table(self.table_name).upsert(rows).execute()

    def fetch_all_documents(self) -> List[Dict[str, Any]]:
        response = self.client.table(self.table_name).select('*').execute()
        return response.data or []

    def count_documents(self) -> int:
        try:
            response = self.client.table(self.table_name).select('id', count='exact').execute()
            return int(response.count or 0)
        except Exception:
            try:
                return len(self.fetch_all_documents())
            except Exception:
                return 0

    def query_documents(self, query: str, embeddings: Any, top_k: int = 5, metadata_filter: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        query_embedding = embeddings.embed_query(query)
        rows = self.fetch_all_documents()
        candidates = []
        for row in rows:
            metadata = row.get('metadata') or {}
            if metadata_filter and not all(str(metadata.get(k, '')).lower() == str(v).lower() for k, v in metadata_filter.items()):
                continue
            embedding_value = self._deserialize_embedding(row.get('embedding'))
            if not embedding_value:
                continue
            doc_vector = np.array(embedding_value, dtype='float32')
            query_vector = np.array(query_embedding, dtype='float32')
            similarity = float(np.dot(doc_vector, query_vector) / max(np.linalg.norm(doc_vector) * np.linalg.norm(query_vector), 1e-12))
            candidates.append({
                'score': similarity,
                'text': row.get('content', ''),
                'meta': metadata,
            })
        candidates.sort(key=lambda item: item['score'], reverse=True)
        return candidates[:top_k]

    def upsert_session_summary(self, session_id: str, summary_text: str, entity_index: dict = None, event_count: int = 0) -> None:
        try:
            self.client.table('agent_sessions').upsert({
                'session_id': session_id,
                'summary_text': summary_text,
                'entity_index': entity_index or {},
                'event_count': event_count,
            }).execute()
        except Exception as e:
            logging.warning(f'Failed to persist session summary: {e}')

    def get_session_summary(self, session_id: str) -> Optional[dict]:
        try:
            res = self.client.table('agent_sessions').select('*').eq('session_id', session_id).maybe_single().execute()
            return res.data
        except Exception:
            return None

    def log_code_execution(self, session_id: str, code: str, success: bool, stdout: str = '', duration_ms: int = 0, blocked_by: str = None) -> None:
        try:
            self.client.table('code_executions').insert({
                'session_id': session_id,
                'code_hash': hashlib.sha256(code.encode()).hexdigest(),
                'code_text': code,
                'success': success,
                'stdout_preview': (stdout or '')[:500],
                'duration_ms': duration_ms,
                'blocked_by': blocked_by,
            }).execute()
        except Exception as e:
            logging.warning(f'Failed to log code execution: {e}')

    def log_tool_call(self, session_id: str, tool_name: str, args_summary: dict, status: str, duration_ms: int = 0, error_msg: str = None) -> None:
        try:
            self.client.table('tool_audit_log').insert({
                'session_id': session_id,
                'tool_name': tool_name,
                'args_summary': args_summary,
                'status': status,
                'duration_ms': duration_ms,
                'error_msg': error_msg,
            }).execute()
        except Exception as e:
            logging.warning(f'Failed to log tool audit: {e}')

    def get_chunk_count(self) -> int:
        try:
            res = self.client.table(self.table_name).select('id', count='exact').execute()
            return int(res.count or 0)
        except Exception:
            return 0

    def _deserialize_embedding(self, value: Any) -> List[float]:
        if value is None:
            return []
        if isinstance(value, list):
            return [float(x) for x in value]
        if isinstance(value, str):
            try:
                return [float(x) for x in json.loads(value)]
            except Exception:
                return []
        return [float(value)]
