"""GGUF Jinja chat formatting with bounded input/output and no template globals."""

from jinja2 import StrictUndefined, TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment


def render_chat_template(
    metadata, messages, tokenizer, add_generation_prompt=True, max_chars=65536, tools=None
):
    template = metadata.get("tokenizer.chat_template")
    if not isinstance(template, str) or not template or len(template) > 20000:
        raise ValueError("Missing or oversized GGUF chat template")
    if not isinstance(messages, list) or not messages or len(messages) > 256:
        raise ValueError("Invalid chat messages")
    if tools is None:
        tools = []
    if (
        not isinstance(tools, list)
        or len(tools) > 16
        or any(not isinstance(tool, dict) for tool in tools)
    ):
        raise ValueError("Invalid chat template tools")
    clean = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant", "tool"}
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("Invalid chat message")
        clean.append({"role": message["role"], "content": message["content"]})
    if sum(len(m["content"]) for m in clean) > max_chars:
        raise ValueError("Chat exceeds input limit")
    env = ImmutableSandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
    env.globals.clear()

    def reject(message):
        raise ValueError(str(message))

    env.globals["raise_exception"] = reject

    def special(key):
        token = metadata.get("tokenizer.ggml." + key + "_token_id")
        if token is None:
            return ""
        if not isinstance(token, int) or not 0 <= token < len(tokenizer.vocabulary):
            raise ValueError("Invalid chat special token")
        return tokenizer.vocabulary[token]

    try:
        rendered = env.from_string(template).render(
            messages=clean,
            add_generation_prompt=add_generation_prompt,
            bos_token=special("bos"),
            eos_token=special("eos"),
            tools=tools,
        )
    except TemplateError as error:
        raise ValueError("Unsupported or unsafe GGUF chat template") from error
    if len(rendered) > max_chars:
        raise ValueError("Rendered chat exceeds limit")
    return rendered
