# Structured output

Use `response_format: {"type": "json_schema", "json_schema": {"name": "result", "strict": true, "schema": ...}}` on completion or chat requests. Supported schemas are the explicit subset in `schema.py`; unsupported constraints return an error. Masks act before sampling filters. A length-limited response can be incomplete and reports `finish_reason: "length"`.

The mask trie shares vocabulary byte prefixes, prunes invalid transitions, and caches the entire immutable grammar stack plus partial UTF-8 bytes. The cache is bounded per request; EOS is allowed only after acceptance with no pending bytes. Special tokens other than EOS are excluded. Recursive stack states are never merged merely because their next character class matches.

Chat uses `tokenizer.chat_template` from GGUF with a restricted Jinja environment. Missing, invalid and unsupported templates fail explicitly. Templates are model metadata and must come from a trusted model source; input and rendered-output limits do not impose an execution-time limit on template evaluation.

Tools currently require `tool_choice: "required"` or a named function choice, produce one constrained `{name, arguments}` object, and translate accepted non-streaming output to `message.tool_calls`. Automatic tool selection and streamed tool deltas are unsupported. Signatures constrain syntax; applications must authorize actual tool execution.

Run `python tools/benchmark_grammar_masks.py`. The recorded benchmark validates four ASCII/Unicode JSON fixtures against independent JSON Schema validation and times mask application. Its logits favor known fixture continuations: it measures mask mechanics, not model semantic quality. Results use the actual execution time regardless of commit dates.
