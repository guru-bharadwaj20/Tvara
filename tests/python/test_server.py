import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from tvara.server import ServerConfig, GenerationEvent, create_app  # noqa: E402


class FixtureBackend:
    """Wire-contract fixture; this class never claims to run model inference."""

    vocab_size = 256
    context_size = 128

    def __init__(self):
        self.closed = False
        self.streams_closed = 0

    def encode(self, text):
        return list(text.encode("utf-8"))

    def render_chat(self, messages):
        return "\n".join(m["role"] + ": " + m["content"] for m in messages) + "\nassistant:"

    def stream(self, prompt, options):
        try:
            for char in "hello"[: options["max_new_tokens"]]:
                yield GenerationEvent(ord(char), char)
            yield GenerationEvent(
                finish_reason="stop" if options["max_new_tokens"] >= 5 else "length"
            )
        finally:
            self.streams_closed += 1

    def close(self):
        self.closed = True


@pytest.fixture
def client():
    backend = FixtureBackend()
    with TestClient(create_app(backend, ServerConfig(served_name="fixture"))) as client:
        yield client
    assert backend.closed


def test_configuration_rejects_invalid_ports():
    with pytest.raises(ValueError):
        ServerConfig(port=0)
    with pytest.raises(ValueError):
        ServerConfig(max_prompt_chars=0)


def test_health_and_readiness_identify_the_fixture_backend(client):
    assert client.get("/health").json() == {"status": "ok", "ready": True}
    assert client.get("/ready").status_code == 200
    assert client.get("/v1/models").json()["data"][0]["id"] == "fixture"


def test_completion_returns_backend_generated_text(client):
    result = client.post(
        "/v1/completions", json={"model": "fixture", "prompt": "abc", "max_tokens": 3}
    )
    assert result.status_code == 200 and result.json()["choices"][0]["text"] == "hel"


def test_chat_contract_and_role_validation(client):
    result = client.post(
        "/v1/chat/completions",
        json={"model": "fixture", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5},
    )
    assert result.status_code == 200 and result.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "hello",
    }
    assert client.post(
        "/v1/chat/completions",
        json={"model": "fixture", "messages": [{"role": "unknown", "content": "Hi"}]},
    ).status_code in (400, 422)


def test_nonstreaming_envelope_records_actual_token_usage(client):
    result = client.post(
        "/v1/completions", json={"model": "fixture", "prompt": "abc", "max_tokens": 3}
    ).json()
    assert (
        result["id"].startswith("cmpl-") and result["model"] == "fixture" and result["created"] > 0
    )
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}
