import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient
from tvara.chat_templates import render_chat_template
from tvara.server import NativeBackend, ServerConfig, GenerationEvent, create_app


class Tokenizer:
    vocabulary = tuple(chr(i) for i in range(128))
    special_ids = frozenset()

    def token_bytes(self, ident):
        return bytes([ident])


TEMPLATE = "{% if tools %}{% for tool in tools %}{{tool.function.name}}:{{tool.function.parameters|tojson}};{% endfor %}{% endif %}{% for message in messages %}{{message.content}}{% endfor %}"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"enum": ["Paris"]}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]


def test_optional_tools_are_defined_when_absent():
    assert (
        render_chat_template(
            {"tokenizer.chat_template": TEMPLATE}, [{"role": "user", "content": "hi"}], Tokenizer()
        )
        == "hi"
    )


def test_native_backend_forwards_declared_signatures():
    backend = object.__new__(NativeBackend)
    backend.model = SimpleNamespace(metadata={"tokenizer.chat_template": TEMPLATE})
    backend.tokenizer = Tokenizer()
    rendered = backend.render_chat([{"role": "user", "content": "hi"}], tools=TOOLS)
    assert "weather:" in rendered and "Paris" in rendered and rendered.endswith("hi")


class TemplateBackend(NativeBackend):
    """Tests request/template wiring only, without claiming model inference."""

    def __init__(self):
        self.model = SimpleNamespace(metadata={"tokenizer.chat_template": TEMPLATE})
        self.tokenizer = Tokenizer()
        self.vocab_size = 128
        self.context_size = 4096
        self.eos_id = None
        self.prompts = []

    def encode(self, text):
        self.prompts.append(text)
        return list(text.encode())

    def stream(self, prompt, options):
        yield GenerationEvent(finish_reason="length")

    def close(self):
        pass


def test_http_passes_validated_tools_into_prompt_and_rejects_empty_declarations():
    backend = TemplateBackend()
    with TestClient(create_app(backend, ServerConfig(served_name="fixture"))) as client:
        request = {
            "model": "fixture",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
            "tools": TOOLS,
            "tool_choice": "required",
        }
        response = client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200, response.text
        assert "weather:" in backend.prompts[-1] and "Paris" in backend.prompts[-1]
        assert client.post("/v1/chat/completions", json={**request, "tools": []}).status_code == 400
        no_tools = {key: value for key, value in request.items() if key != "tools"}
        assert client.post("/v1/chat/completions", json=no_tools).status_code == 400


def test_unsupported_variables_still_fail_strictly():
    with pytest.raises(ValueError):
        render_chat_template(
            {"tokenizer.chat_template": "{{unsupported_global}}"},
            [{"role": "user", "content": "hi"}],
            Tokenizer(),
        )
