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
