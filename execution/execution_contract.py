from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    QUEUED = "queued"
    ACCEPTED = "accepted"
    RUNNING = "running"
    PARTIAL = "partial"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ExecutionMode(str, Enum):
    AUTO = "auto"
    PLAN = "plan"
    EXECUTE = "execute"
    REPORT = "report"
    GRAPH = "graph"


class ExecutionArtifact(BaseModel):
    id: str
    type: str = "artifact"
    filename: str
    storage_path: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None


class ExecutionNodeStatus(BaseModel):
    node_id: str
    type: str
    status: ExecutionStatus = ExecutionStatus.QUEUED
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionInput(BaseModel):
    prompt: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)


class ExecutionOutput(BaseModel):
    answer: Optional[str] = None
    summary: Optional[str] = None
    artifacts: List[ExecutionArtifact] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class ExecutionRecord(BaseModel):
    execution_id: str
    tenant_id: str
    user_id: str
    workflow_id: Optional[str] = None
    agent_id: str
    mode: ExecutionMode = ExecutionMode.AUTO
    status: ExecutionStatus = ExecutionStatus.QUEUED
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    input: ExecutionInput = Field(default_factory=ExecutionInput)
    output: ExecutionOutput = Field(default_factory=ExecutionOutput)
    artifacts: List[ExecutionArtifact] = Field(default_factory=list)
    trace_id: Optional[str] = None
    node_status: List[ExecutionNodeStatus] = Field(default_factory=list)
    error: Optional[str] = None
    approved_by: Optional[str] = None
    permissions: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def create_execution_record(
    *,
    tenant_id: str,
    user_id: str,
    agent_id: str,
    mode: ExecutionMode = ExecutionMode.AUTO,
    workflow_id: Optional[str] = None,
    prompt: Optional[str] = None,
    attachments: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> ExecutionRecord:
    timestamp = datetime.now(timezone.utc).isoformat()
    return ExecutionRecord(
        execution_id=f"exec_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        tenant_id=tenant_id,
        user_id=user_id,
        workflow_id=workflow_id,
        agent_id=agent_id,
        mode=mode,
        status=ExecutionStatus.QUEUED,
        created_at=timestamp,
        updated_at=timestamp,
        started_at=None,
        completed_at=None,
        input=ExecutionInput(
            prompt=prompt,
            attachments=attachments or [],
            parameters=parameters or {},
        ),
        output=ExecutionOutput(),
        artifacts=[],
        trace_id=trace_id or f"trace_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        node_status=[],
        error=None,
        approved_by=None,
        permissions={},
        metadata=metadata or {},
    )


def is_terminal_status(status: ExecutionStatus) -> bool:
    return status in {ExecutionStatus.FAILED, ExecutionStatus.COMPLETED, ExecutionStatus.CANCELLED}
