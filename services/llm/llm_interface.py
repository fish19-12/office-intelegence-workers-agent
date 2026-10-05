from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallResponse:
    type: str              # "tool_call" | "text"
    text: str = ""
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)


class BaseLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, **kwargs) -> Any:
        """Generate and parse JSON output from the LLM. Returns Python object or raises ValueError."""
        raw = self.generate(prompt, **kwargs)
        try:
            return json.loads(raw)
        except Exception:
            # Try to extract a JSON substring
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end+1])
                except Exception:
                    pass
            # Try array
            start = raw.find('[')
            end = raw.rfind(']')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end+1])
                except Exception:
                    pass
        raise ValueError(f"LLM did not return parseable JSON. Raw:\n{raw}")

    def generate_with_tools(self, prompt: str, tools: List[dict], messages: List[dict] | None = None) -> ToolCallResponse:
        augmented = self._augment_prompt_with_tools(prompt, tools)
        text = self.generate(augmented)
        return self._parse_action_tool_call(text)

    def _augment_prompt_with_tools(self, prompt: str, tools: List[dict]) -> str:
        tool_desc = "\n".join(
            f'- {t["function"]["name"]}: {t["function"]["description"]}'
            for t in tools
        )
        return (
            f"{prompt}\n\nAvailable tools:\n{tool_desc}\n\n"
            "If you need a tool, respond with exactly:\n"
            "ACTION: tool_name {\"arg\": \"value\"}\n"
            "Otherwise answer directly."
        )

    def _parse_action_tool_call(self, text: str) -> ToolCallResponse:
        content = text.strip()
        if content.startswith("ACTION:"):
            payload = content[7:].strip()
            parts = payload.split(" ", 1)
            name = parts[0] if parts else None
            args = {}
            if len(parts) > 1:
                try:
                    args = json.loads(parts[1])
                except json.JSONDecodeError:
                    args = {}
            return ToolCallResponse(
                type="tool_call",
                text="",
                tool_name=name,
                tool_args=args,
                raw_response={},
            )
        return ToolCallResponse(
            type="text",
            text=content,
            tool_name=None,
            tool_args={},
            raw_response={},
        )


class MockLLM(BaseLLM):
    def generate(self, prompt: str, **kwargs) -> str:
        header = "[MOCK LLM RESPONSE]\n"
        return header + prompt[:400]

    def invoke(self, messages, **kwargs):
        text = ""
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, tuple) and len(msg) == 2:
                    _, payload = msg
                    text += str(payload)
                elif isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content")
                    if content is not None:
                        text += f"{role}: {content}\n"
                else:
                    text += str(msg)
        else:
            text = str(messages)
        return type("MockResponse", (), {"content": self.generate(text, **kwargs)})()

    def generate_with_tools(self, prompt: str, tools: List[dict], messages: List[dict] | None = None) -> ToolCallResponse:
        return super().generate_with_tools(prompt, tools, messages)


class OpenAIAdapter(BaseLLM):
    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError('openai package is required for OpenAIAdapter') from e
        self.model = config.get('model', 'gpt-3.5-turbo')
        api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        api_base = config.get('api_base') or os.environ.get('OPENAI_API_BASE')
        
        client_kwargs = {'api_key': api_key}
        if api_base:
            client_kwargs['base_url'] = api_base
        self.client = OpenAI(**client_kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=kwargs.get('max_tokens', 256),
                temperature=kwargs.get('temperature', 0.2),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f'[OpenAIAdapter] Error: {str(e)}'

    def generate_with_tools(self, prompt: str, tools: List[dict], messages: List[dict] | None = None) -> ToolCallResponse:
        messages = list(messages) if messages is not None else []
        messages.append({'role': 'user', 'content': prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice='auto',
                temperature=0.1,
            )
            msg = response.choices[0].message
            tool_calls = getattr(msg, 'tool_calls', None)
            raw = getattr(response, 'model_dump', lambda: {})()
            if tool_calls:
                call = tool_calls[0]
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                return ToolCallResponse(
                    type='tool_call',
                    text='',
                    tool_name=call.function.name,
                    tool_args=args,
                    raw_response=raw,
                )
            return ToolCallResponse(
                type='text',
                text=(getattr(msg, 'content', '') or '').strip(),
                tool_name=None,
                tool_args={},
                raw_response=raw,
            )
        except Exception as e:
            return ToolCallResponse(type='text', text=f'[OpenAIAdapter] Error: {e}', tool_name=None, tool_args={}, raw_response={})


class HuggingFaceAdapter(BaseLLM):
    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError('transformers package is required for HuggingFaceAdapter') from e
        model_name = config.get('model_name', 'gpt2')
        self.generator = pipeline(
            'text-generation',
            model=model_name,
            device=config.get('device', -1),
            trust_remote_code=config.get('trust_remote_code', False),
        )
        self.max_length = config.get('max_length', 256)

    def generate(self, prompt: str, **kwargs) -> str:
        result = self.generator(prompt, max_length=kwargs.get('max_length', self.max_length), do_sample=kwargs.get('do_sample', False))
        return result[0]['generated_text'].strip()

    def generate_with_tools(self, prompt: str, tools: List[dict], messages: List[dict] | None = None) -> ToolCallResponse:
        return super().generate_with_tools(prompt, tools, messages)


class DeepseekAdapter(BaseLLM):
    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        endpoint = config.get('endpoint') or os.environ.get('DEEPSEEK_API_URL')
        api_base = config.get('api_base') or os.environ.get('DEEPSEEK_API_BASE') or os.environ.get('OPENAI_API_BASE')
        api_key = config.get('api_key') or os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if endpoint:
            self.impl = GenericHTTPAdapter({'endpoint': endpoint, 'api_key': api_key, 'payload_key': config.get('payload_key', 'prompt')})
        elif api_base:
            openai_config = {'api_key': api_key, 'api_base': api_base, 'model': config.get('model', 'deepseek-chat')}
            self.impl = OpenAIAdapter(openai_config)
        else:
            try:
                self.impl = OpenAIAdapter(config)
            except Exception:
                self.impl = HuggingFaceAdapter(config)

    def generate(self, prompt: str, **kwargs) -> str:
        return self.impl.generate(prompt, **kwargs)

    def generate_with_tools(self, prompt: str, tools: List[dict], messages: List[dict] | None = None) -> ToolCallResponse:
        return self.impl.generate_with_tools(prompt, tools, messages)


class GeminiAdapter(BaseLLM):
    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        endpoint = config.get('endpoint') or os.environ.get('GEMINI_API_URL')
        api_key = config.get('api_key') or os.environ.get('GEMINI_API_KEY')
        if endpoint:
            self.impl = GenericHTTPAdapter({'endpoint': endpoint, 'api_key': api_key, 'payload_key': config.get('payload_key', 'prompt')})
        else:
            self.impl = HuggingFaceAdapter(config)

    def generate(self, prompt: str, **kwargs) -> str:
        return self.impl.generate(prompt, **kwargs)

    def generate_with_tools(self, prompt: str, tools: List[dict], messages: List[dict] | None = None) -> ToolCallResponse:
        return self.impl.generate_with_tools(prompt, tools, messages)


class LLMFactory:
    @staticmethod
    def create(name: str, config: Dict[str, Any] | None = None) -> BaseLLM:
        # Support two invocation styles:
        # 1) LLMFactory.create('openai', config)
        # 2) LLMFactory.create(config_dict)  -> config_dict may include 'llm_provider' or 'provider'
        if isinstance(name, dict):
            cfg = name
            provider = (cfg.get('llm_provider') or cfg.get('provider') or 'deepseek').lower()
            config = cfg
            name = provider
        else:
            name = (name or 'deepseek').lower()

        if name == 'mock':
            return MockLLM()

        api_key = config.get('api_key') if isinstance(config, dict) else None
        env_api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('LLM_API_KEY')

        if name in ('openai', 'chatgpt'):
            if not (api_key or env_api_key):
                return MockLLM()
            return OpenAIAdapter(config)
        if name in ('huggingface', 'hf'):
            return HuggingFaceAdapter(config)
        if name == 'deepseek':
            if not (api_key or env_api_key):
                return MockLLM()
            return DeepseekAdapter(config)
        if name == 'gemini':
            if not (config.get('api_key') or os.environ.get('GEMINI_API_KEY')):
                return MockLLM()
            return GeminiAdapter(config)
        return MockLLM()


class GenericHTTPAdapter(BaseLLM):
    """Simple HTTP-backed adapter for arbitrary LLM HTTP endpoints.

    Expects a JSON POST where the prompt is sent under `payload_key` (default `prompt`).
    Configure via `endpoint` and `api_key` in `config` or env vars `LLM_API_URL`/`LLM_API_KEY`.
    """
    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        self.endpoint = config.get('endpoint') or os.environ.get('LLM_API_URL')
        self.api_key = config.get('api_key') or os.environ.get('LLM_API_KEY')
        self.payload_key = config.get('payload_key', 'prompt')
        self.extra_payload = config.get('extra_payload', {})
        self.headers = config.get('headers', {})
        if self.api_key:
            self.headers.setdefault('Authorization', f'Bearer {self.api_key}')
        try:
            import requests
        except Exception as e:
            raise ImportError('requests is required for GenericHTTPAdapter') from e
        self.requests = requests

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.endpoint:
            return '[GenericHTTPAdapter] No endpoint configured.'
        body = {self.payload_key: prompt}
        body.update(self.extra_payload)
        try:
            resp = self.requests.post(self.endpoint, json=body, headers=self.headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # common response fields
            for key in ('text', 'output', 'result', 'response'):
                if isinstance(data, dict) and key in data:
                    return data[key]
            # fallback: try to stringify entire JSON
            return str(data)
        except Exception as e:
            return f'[GenericHTTPAdapter] request error: {e}'

    def generate_with_tools(self, prompt: str, tools: List[dict], messages: List[dict] | None = None) -> ToolCallResponse:
        return super().generate_with_tools(prompt, tools, messages)
