import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integrations.mcp_manager import MCPError, MCPManager, ToolRegistry
from integrations.tools import ToolBase, ToolCredentialStore


class RequiredCredentialTool(ToolBase):
    """Tool that requires credentials to be configured."""

    args_schema = {
        "type": "object",
        "required": ["input"],
        "properties": {
            "input": {"type": "string"},
        },
    }

    credential_requirements = [
        {
            "env": "TEST_API_KEY",
            "label": "Test API key",
            "required": True,
        },
        {
            "env": "TEST_SECRET",
            "label": "Test secret",
            "required": True,
            "secret": True,
        },
    ]

    def __init__(self):
        super().__init__(
            name="required_credential_tool",
            description="A tool that requires credentials",
            cooldown=0,
        )

    def execute(self, input: str, **kwargs):
        api_key = self.get_credential("TEST_API_KEY")
        secret = self.get_credential("TEST_SECRET")
        if not api_key or not secret:
            raise ValueError("Credentials not found")
        return {"result": f"Processed: {input} with {api_key}"}


class OptionalCredentialTool(ToolBase):
    """Tool that optionally requires credentials."""

    args_schema = {
        "type": "object",
        "required": ["input"],
        "properties": {
            "input": {"type": "string"},
        },
    }

    credential_requirements = [
        {
            "env": "OPTIONAL_KEY",
            "label": "Optional credential",
            "required": False,
        },
    ]

    def __init__(self):
        super().__init__(
            name="optional_credential_tool",
            description="A tool with optional credentials",
            cooldown=0,
        )

    def execute(self, input: str, **kwargs):
        optional = self.get_credential("OPTIONAL_KEY")
        return {"result": f"Processed: {input}", "has_optional": bool(optional)}


def test_tool_credential_store_save_and_load():
    """Test persisting and loading credentials."""
    # Clear store first
    ToolCredentialStore.FILE_PATH.unlink(missing_ok=True)

    test_data = {"TEST_KEY": "test_value", "ANOTHER_KEY": "another_value"}

    ToolCredentialStore.update(test_data)
    loaded = ToolCredentialStore.load()

    assert loaded["TEST_KEY"] == "test_value"
    assert loaded["ANOTHER_KEY"] == "another_value"


def test_tool_is_not_configured_when_credentials_missing():
    """Test that tool reports not configured when required credentials are missing."""
    # Clear store first
    ToolCredentialStore.FILE_PATH.unlink(missing_ok=True)

    tool = RequiredCredentialTool()

    assert not tool.is_configured()
    assert set(tool.missing_credentials()) >= {"TEST_API_KEY", "TEST_SECRET"}


def test_tool_is_configured_when_credentials_present_in_store():
    """Test that tool reports configured when credentials are in credential store."""
    # Clear store first
    ToolCredentialStore.FILE_PATH.unlink(missing_ok=True)

    ToolCredentialStore.update({
        "TEST_API_KEY": "key123",
        "TEST_SECRET": "secret456",
    })

    tool = RequiredCredentialTool()
    assert tool.is_configured()
    assert tool.missing_credentials() == []
    assert tool.get_credential("TEST_API_KEY") == "key123"


def test_mcp_manager_blocks_execution_of_unconfigured_tool():
    """Test that MCPManager rejects tool execution if not configured."""
    # Clear store first
    ToolCredentialStore.FILE_PATH.unlink(missing_ok=True)

    manager = MCPManager(registry=ToolRegistry())
    manager.registry.register(RequiredCredentialTool())

    with pytest.raises(MCPError) as exc_info:
        manager.call_tool("required_credential_tool", input="test")

    assert "not configured" in str(exc_info.value).lower()
    assert "Missing credentials" in str(exc_info.value)


def test_mcp_manager_allows_execution_of_configured_tool():
    """Test that MCPManager allows execution when credentials are configured."""
    # Clear store first
    ToolCredentialStore.FILE_PATH.unlink(missing_ok=True)

    ToolCredentialStore.update({
        "TEST_API_KEY": "key789",
        "TEST_SECRET": "secret012",
    })

    manager = MCPManager(registry=ToolRegistry())
    manager.registry.register(RequiredCredentialTool())

    result = manager.call_tool("required_credential_tool", input="hello")

    assert result["result"] == "Processed: hello with key789"


def test_tool_credential_details_exposed_in_registry():
    """Test that credential requirements are exposed in tool registry list."""
    # Clear store first
    ToolCredentialStore.FILE_PATH.unlink(missing_ok=True)

    manager = MCPManager(registry=ToolRegistry())
    manager.registry.register(RequiredCredentialTool())

    registry_list = manager.registry.list()
    tool_info = registry_list["required_credential_tool"]

    assert "credential_requirements" in tool_info
    assert isinstance(tool_info["credential_requirements"], list)
    assert len(tool_info["credential_requirements"]) == 2
    assert tool_info["configured"] is False
    assert "TEST_API_KEY" in tool_info["missing_credentials"]


def test_optional_credentials_do_not_block_execution():
    """Test that optional credentials do not prevent tool execution."""
    # Clear store first
    ToolCredentialStore.FILE_PATH.unlink(missing_ok=True)

    manager = MCPManager(registry=ToolRegistry())
    manager.registry.register(OptionalCredentialTool())

    # Even without the optional credential, tool should be configured
    tool = OptionalCredentialTool()
    assert tool.is_configured()

    result = manager.call_tool("optional_credential_tool", input="data")
    assert result["result"] == "Processed: data"
    assert result["has_optional"] is False
