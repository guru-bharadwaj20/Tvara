"""CPU inference HTTP adapter with explicit backend ownership."""

from .response_constraints import tool_response_format

from .chat_templates import render_chat_template

from .response_constraints import response_constraint

from fastapi.exceptions import RequestValidationError

import anyio

import json
from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

import time
import uuid

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerConfig:
    served_name: str = "tvara-local"
    host: str = "127.0.0.1"
    port: int = 8000
    max_prompt_chars: int = 65536
    max_new_tokens: int = 512

    def __post_init__(self):
        if (
            not self.served_name
            or not 1 <= self.port <= 65535
            or self.max_prompt_chars < 1
            or self.max_new_tokens < 0
        ):
            raise ValueError("Invalid server configuration")


@dataclass(frozen=True)
class GenerationEvent:
    token: int | None = None
    text: str = ""
    finish_reason: str | None = None


class NativeBackend:
    def __init__(self, model, tokenizer):
        model.attach_tokenizer(tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = model.vocab_size
        self.context_size = model.context_size
        self.eos_id = model.metadata.get("tokenizer.ggml.eos_token_id")

    def encode(self, text):
        return self.model.encode(text, allow_special=True)

    def render_chat(self, messages):
        return render_chat_template(self.model.metadata, messages, self.tokenizer)

    def stream(self, prompt, options):
        from .tokenizer import TokenTextStream

        decoder = TokenTextStream(self.tokenizer, skip_special=True)
        reason = "length"
        iterator = self.model.iter_tokens(prompt, eos_id=self.eos_id, **options)
        try:
            for token in iterator:
                yield GenerationEvent(token, decoder.feed([token]))
                if token == self.eos_id:
                    reason = "stop"
            tail = decoder.feed([], final=True)
            if tail:
                yield GenerationEvent(text=tail)
            yield GenerationEvent(finish_reason=reason)
        finally:
            iterator.close()

    def close(self):
        self.model.close()


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str | list[int]
    max_tokens: int = Field(default=16, ge=0)
    temperature: float = Field(default=0, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1, gt=0, le=1, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0, le=18446744073709551615)
    stream: bool = False
    response_format: dict | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    max_tokens: int = Field(default=16, ge=0)
    temperature: float = Field(default=0, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1, gt=0, le=1, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0, le=18446744073709551615)
    stream: bool = False
    response_format: dict | None = None


# REQUEST_TYPES
def collect_output(backend, ids, options):
    text = []
    tokens = 0
    reason = "length"
    for event in backend.stream(ids, options):
        text.append(event.text)
        if event.token is not None:
            tokens += 1
        if event.finish_reason:
            reason = event.finish_reason
    return {"text": "".join(text), "tokens": tokens, "finish_reason": reason}


def response_metadata(config, chat=False):
    return {
        "id": ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex,
        "object": "chat.completion" if chat else "text_completion",
        "created": int(time.time()),
        "model": config.served_name,
    }


def sse(data):
    return "data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n\n"


async def stream_events(backend, ids, options, metadata, chat=False, request=None):
    metadata = dict(metadata)
    if chat:
        metadata["object"] = "chat.completion.chunk"
        yield sse(
            {
                **metadata,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
    iterator = backend.stream(ids, options)
    try:
        async for event in iterate_in_threadpool(iterator):
            if request is not None and await request.is_disconnected():
                return
            choice = {"index": 0, "finish_reason": event.finish_reason}
            if chat:
                choice["delta"] = {"content": event.text} if event.text else {}
            else:
                choice.update(text=event.text, logprobs=None)
            yield sse({**metadata, "choices": [choice]})
    except (RuntimeError, ValueError):
        yield sse(
            {
                "error": {
                    "message": "Inference failed",
                    "type": "server_error",
                    "code": "inference_error",
                }
            }
        )
    finally:
        with anyio.CancelScope(shield=True):
            if hasattr(iterator, "close"):
                await run_in_threadpool(iterator.close)
    yield "data: [DONE]\n\n"
    # STREAM_DONE


# RESPONSE_HELPERS
def create_app(backend, config=None):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    config = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            if hasattr(backend, "close"):
                backend.close()

    app = FastAPI(title="Tvara", lifespan=lifespan)
    app.state.backend = backend
    app.state.config = config
    app.state.ready = False

    @app.get("/health")
    async def health():
        return {"status": "ok", "ready": app.state.ready}

    @app.get("/ready")
    async def ready():
        from fastapi.responses import JSONResponse

        return JSONResponse({"ready": app.state.ready}, status_code=200 if app.state.ready else 503)

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {"id": config.served_name, "object": "model", "created": 0, "owned_by": "local"}
            ],
        }

    async def complete(body, chat=False, request=None):
        if body.model != config.served_name:
            raise ValueError("Unknown model")
        if body.max_tokens > config.max_new_tokens:
            raise ValueError("Generation limit exceeds server configuration")
        prompt = (
            backend.render_chat([m.model_dump() for m in body.messages]) if chat else body.prompt
        )
        if isinstance(prompt, str) and len(prompt) > config.max_prompt_chars:
            raise ValueError("Prompt exceeds configured character limit")
        ids = list(prompt) if isinstance(prompt, list) else backend.encode(prompt)
        if not ids or any(
            not isinstance(token, int) or not 0 <= token < backend.vocab_size for token in ids
        ):
            raise ValueError("Invalid prompt token IDs")
        if len(ids) + body.max_tokens > backend.context_size:
            raise ValueError("Request exceeds model context")
        options = {
            "max_new_tokens": body.max_tokens,
            "temperature": body.temperature,
            "seed": body.seed,
            "top_p": body.top_p,
        }
        if chat and body.tools:
            if body.response_format is not None:
                raise ValueError("Tools and response_format cannot be combined")
            if body.stream:
                raise ValueError("Streaming tool calls are not supported")
            body.response_format = tool_response_format(body.tools, body.tool_choice)
        if body.response_format is not None:
            if not hasattr(backend, "tokenizer"):
                raise ValueError("Backend lacks tokenizer for structured output")
            options["token_constraint"] = response_constraint(
                body.response_format, backend.tokenizer, getattr(backend, "eos_id", None)
            )
        if body.stream:
            return StreamingResponse(
                stream_events(
                    backend, ids, options, response_metadata(config, chat), chat, request
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        result = await run_in_threadpool(collect_output, backend, ids, options)
        return {
            **response_metadata(config, chat),
            "choices": [
                {
                    "index": 0,
                    "text": result["text"],
                    "finish_reason": result["finish_reason"],
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": result["tokens"],
                "total_tokens": len(ids) + result["tokens"],
            },
        }

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        return await complete(body, request=request)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest, request: Request):
        result = await complete(body, chat=True, request=request)
        if body.stream:
            return result
        choice = result["choices"][0]
        choice["message"] = {"role": "assistant", "content": choice.pop("text")}
        if body.tools and choice["finish_reason"] == "stop":
            call = json.loads(choice["message"]["content"])
            choice["message"] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_" + uuid.uuid4().hex,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                        },
                    }
                ],
            }
            choice["finish_reason"] = "tool_calls"
        return result

    @app.exception_handler(ValueError)
    async def invalid_request(request, error):
        return JSONResponse(
            {
                "error": {
                    "message": str(error),
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                }
            },
            status_code=400,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse(
            {
                "error": {
                    "message": "Invalid request fields",
                    "type": "invalid_request_error",
                    "code": "validation_error",
                }
            },
            status_code=400,
        )

    @app.exception_handler(RuntimeError)
    async def inference_error(request, error):
        return JSONResponse(
            {
                "error": {
                    "message": "Inference failed",
                    "type": "server_error",
                    "code": "inference_error",
                }
            },
            status_code=503,
        )

    # ROUTES
    return app


def main():
    parser = argparse.ArgumentParser(description="Serve a validated native GGUF model on CPU")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--name", default="tvara-local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from .native import NativeModel
    from .tokenizer import ByteBPE
    import uvicorn

    config = ServerConfig(served_name=args.name, host=args.host, port=args.port)
    model = NativeModel(args.model)
    try:
        backend = NativeBackend(model, ByteBPE.from_json(args.tokenizer))
        uvicorn.run(create_app(backend, config), host=config.host, port=config.port)
    finally:
        model.close()


if __name__ == "__main__":
    main()
