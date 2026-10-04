"""CPU inference HTTP adapter with explicit backend ownership."""

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
        raise ValueError("A validated chat template is required for native chat")

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


# REQUEST_TYPES
# RESPONSE_HELPERS
def create_app(backend, config=None):
    from fastapi import FastAPI

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
