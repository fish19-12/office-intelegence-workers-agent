import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integrations.mcp_manager import MCPError, MCPManager, ToolRegistry
from integrations.tools import ToolBase


class FailingTool(ToolBase):
    args_schema = {
        'type': 'object',
        'required': ['value'],
        'properties': {
            'value': {'type': 'integer'},
        },
    }

    def __init__(self):
        super().__init__(name='failing_tool', description='A tool that always fails', cooldown=0)

    def execute(self, value: int, **kwargs):
        raise ValueError('boom')


class EchoTool(ToolBase):
    args_schema = {
        'type': 'object',
        'required': ['text'],
        'properties': {
            'text': {'type': 'string'},
        },
    }

    def __init__(self):
        super().__init__(name='echo_tool', description='Echo text back', cooldown=0)

    def execute(self, text: str, **kwargs):
        self.update_last_used()
        return {'echo': text}


def test_mcp_manager_rejects_invalid_tool_args_with_schema_error():
    manager = MCPManager(registry=ToolRegistry())
    manager.registry.register(FailingTool())

    with pytest.raises(MCPError, match=r'Invalid args') as exc_info:
        manager.call_tool('failing_tool', invalid=123)

    assert 'value' in str(exc_info.value)


def test_mcp_manager_wraps_tool_execution_exceptions_in_mcp_error():
    manager = MCPManager(registry=ToolRegistry())
    manager.registry.register(FailingTool())

    with pytest.raises(MCPError) as exc_info:
        manager.call_tool('failing_tool', value=1)

    message = str(exc_info.value)
    assert 'ValueError' in message
    assert 'boom' in message
    assert manager._audit_log, 'Expected audit log entry for failed tool execution'
    last_entry = manager._audit_log[-1]
    assert last_entry['status'] == 'failed'
    assert 'error' in last_entry
    assert 'failing_tool' == last_entry['tool']


def test_mcp_manager_records_successful_tool_execution_in_audit_log():
    manager = MCPManager(registry=ToolRegistry())
    manager.registry.register(EchoTool())

    result = manager.call_tool('echo_tool', text='hello')

    assert result == {'echo': 'hello'}
    assert manager._audit_log, 'Expected audit log entry for successful tool execution'
    last_entry = manager._audit_log[-1]
    assert last_entry['status'] == 'success'
    assert last_entry['tool'] == 'echo_tool'
    assert last_entry['args_summary']['text'] == 'hello'


def test_mcp_manager_flushes_queued_tool_after_cooldown():
    manager = MCPManager(registry=ToolRegistry())
    tool = EchoTool()
    tool.cooldown = 2
    manager.registry.register(tool)

    first_result = manager.call_tool('echo_tool', text='first')
    assert first_result == {'echo': 'first'}

    queued_result = manager.call_tool('echo_tool', text='second')
    assert queued_result['status'] == 'queued'
    assert queued_result['queued'] is True

    tool.last_used_at -= 3
    flushed = manager.flush_queue()

    assert len(flushed) == 1
    assert flushed[0]['tool'] == 'echo_tool'
    assert flushed[0]['result'] == {'echo': 'second'}
    assert any(entry['status'] == 'queued' for entry in manager._audit_log)
    assert any(entry['status'] == 'success' for entry in manager._audit_log)


def test_mcp_manager_route_agent_is_deprecated():
    manager = MCPManager(registry=ToolRegistry())

    with pytest.raises(MCPError, match=r'deprecated'):
        manager.route_agent('analytics', 'execute_python', code='print("route ok")', timeout=5)
