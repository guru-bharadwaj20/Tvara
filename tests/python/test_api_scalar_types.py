"""JSON request types must not coerce booleans into token IDs or counts."""

import pytest
from fastapi.testclient import TestClient
from tvara.server import GenerationEvent, ServerConfig, create_app


class Backend:
    vocab_size = 3
    context_size = 64

    def __init__(self):
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return [0]

    def render_chat(self, messages):
        self.calls += 1
        return "chat"

    def stream(self, prompt, options):
        self.calls += 1
        yield GenerationEvent(1, "x")
        yield GenerationEvent(finish_reason="length")

    def close(self):
        pass


@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", True),
        ("max_tokens", "1"),
        ("temperature", True),
        ("temperature", "1.0"),
        ("top_p", True),
        ("seed", True),
        ("seed", 1.5),
        ("stream", "false"),
    ],
)
def test_invalid_json_scalar_types_return_validation_error_before_inference(endpoint, field, value):
    backend = Backend()
    body = {"model": "fixture", "max_tokens": 1, "temperature": 0}
    body.update(
        {"messages": [{"role": "user", "content": "Hi"}]}
        if endpoint.endswith("chat/completions")
        else {"prompt": "Hi"}
    )
    body[field] = value
    with TestClient(create_app(backend, ServerConfig(served_name="fixture"))) as client:
        response = client.post(endpoint, json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"
    assert backend.calls == 0


def test_boolean_prompt_token_id_is_rejected_before_inference():
    backend = Backend()
    with TestClient(create_app(backend, ServerConfig(served_name="fixture"))) as client:
        response = client.post(
            "/v1/completions", json={"model": "fixture", "prompt": [True], "max_tokens": 1}
        )
    assert response.status_code == 400 and backend.calls == 0


@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
def test_integer_json_values_for_float_sampling_controls_remain_valid(endpoint):
    backend = Backend()
    body = {
        "model": "fixture",
        "max_tokens": 1,
        "temperature": 0,
        "top_p": 1,
        "seed": 1,
        "stream": False,
    }
    body.update(
        {"messages": [{"role": "user", "content": "Hi"}]}
        if endpoint.endswith("chat/completions")
        else {"prompt": "Hi"}
    )
    with TestClient(create_app(backend, ServerConfig(served_name="fixture"))) as client:
        response = client.post(endpoint, json=body)
    assert response.status_code == 200 and backend.calls > 0
