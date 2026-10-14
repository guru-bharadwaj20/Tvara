# Structured output

Use `response_format: {"type": "json_schema", "json_schema": {"name": "result", "strict": true, "schema": ...}}` on completion or chat requests. Supported schemas are the explicit subset in `schema.py`; unsupported constraints return an error. Masks act before sampling filters. A length-limited response can be incomplete and reports `finish_reason: "length"`.

The mask trie shares vocabulary byte prefixes, prunes invalid transitions, and caches the entire immutable grammar stack plus partial UTF-8 bytes. The cache is bounded per request; EOS is allowed only after acceptance with no pending bytes. Special tokens other than EOS are excluded. Recursive stack states are never merged merely because their next character class matches.

Chat uses `tokenizer.chat_template` from GGUF with a restricted Jinja environment. Missing, invalid and unsupported templates fail explicitly. Templates are model metadata and must come from a trusted model source; input and rendered-output limits do not impose an execution-time limit on template evaluation.

Tools currently require `tool_choice: "required"` or a named function choice, produce one constrained `{name, arguments}` object, and translate accepted non-streaming output to `message.tool_calls`. Automatic tool selection and streamed tool deltas are unsupported. Signatures constrain syntax; applications must authorize actual tool execution.

Run `python tools/benchmark_grammar_masks.py`. The recorded benchmark validates four ASCII/Unicode JSON fixtures against independent JSON Schema validation and times mask application. Its logits favor known fixture continuations: it measures mask mechanics, not model semantic quality. Results use the actual execution time regardless of commit dates.

Run `python tools/benchmark_grammar_masks.py --compare --output bench/results/grammar-mask-comparison.json` for the naive-scan comparison. All three paths must produce identical allowed IDs before timing is recorded. Fifteen local repetitions cover a synthetic 4,096-token shared-prefix vocabulary and one state. The cache-hit path includes unpacking IDs; this is a scoped microbenchmark, not an end-to-end generation speedup.

## Final compliance evaluation

`python tools/evaluate_structured_output.py` sends 252 strict `json_schema` requests through the FastAPI application and the native C ABI backend: twelve schemas covering the supported subset (nested objects, enums with non-ASCII values, `const`, bounded arrays and strings, `email`/`uuid` formats, `anyOf` and type lists, `$defs`/`$ref`, min/max properties, a top-level array, an unbounded string and a two-function tool call), each with greedy decoding and twenty seeded temperature-1.0 samples (top_p 1.0 and 0.9). A finished response (`finish_reason: "stop"`) must parse and pass an independent `jsonschema` Draft 2020-12 validator with format checking. A length-limited response is re-encoded and replayed through a fresh constraint; every byte must be accepted. Every third request is repeated with `stream: true` and must reproduce the same text and finish reason. Eight unsupported requests (`pattern`, `minimum`, `oneOf`, external `$ref`, oversized string bounds, unknown format, a false schema, `strict: false`) must fail with HTTP 400 before generation.

Recorded result (`bench/results/structured-output-compliance.json`, measured 2026-10-14 UTC on the 32-thread host B listed in [performance.md](performance.md)): 235 of 235 finished responses were schema-valid, all 17 length-limited responses were valid prefixes, 84 of 84 streaming checks matched, 8 of 8 unsupported schemas were rejected and no request failed. Every greedy request hit the 192-token limit because the random model's highest-scoring allowed token was whitespace, which the JSON `ws` rule permits without bound; finish rates therefore describe this fixture, not a trained model.

The first full run found a defect: Python's incremental UTF-8 decoder accepts `ED A0`–`ED BF` as a pending prefix and only rejects the surrogate at the third byte, so `pending_can_match` raised `UnicodeDecodeError` and the request returned HTTP 400 mid-generation (4 of 252 requests). Undecodable pending prefixes are now pruned from the mask, covered by `test_surrogate_prefixes_are_pruned_instead_of_raising`.

Limits: random weights, a 257-token byte vocabulary and a 512-token context. The evaluation measures constraint enforcement, not whether a trained model selects semantically correct values, and chat-template rendering is exercised separately by the demo in [demo.md](demo.md).
