# Sampling design

Native code returns float32 logits; token selection happens in Python for every
served path. See [architecture.md](architecture.md) for the surrounding flow.

## Where sampling lives

| Path | Code | Controls |
|---|---|---|
| HTTP completions and chat | [server.py](../python/tvara/server.py) into `NativeModel.iter_tokens` | `temperature`, `top_p`, `seed`, `max_tokens`, `response_format`, tools |
| Python API | `iter_tokens` in [native.py](../python/tvara/native.py) | also `top_k`, `min_p`, `repetition`, `frequency`, `presence`, `eos_id`, `token_constraint` |
| C++ | `generate` in [generation.hpp](../core/include/tvara/generation.hpp), [sampling.hpp](../core/include/tvara/sampling.hpp) | same filters and penalties, no grammar; contiguous executor, tests only |
| Scheduler backend | `NativeSchedulerBackend` in [scheduler.py](../python/tvara/scheduler.py) | greedy only |
| Speculation | `NativeSpeculativeModel` in [speculative.py](../python/tvara/speculative.py) | temperature only |

The HTTP layer does not expose `top_k`, `min_p` or penalties; they keep their
neutral defaults there.

## Pipeline order

`iter_tokens` processes each step in float64:

1. Reject nonfinite logits.
2. Penalties over prompt plus generated history: for each seen token, divide a
   nonnegative logit by `repetition` or multiply a negative one by it, then
   subtract `frequency * count + presence`.
3. Grammar mask, if a `token_constraint` is set: disallowed logits become `-inf`.
4. `temperature == 0`: argmax (lowest index on ties). Filters are skipped.
5. Otherwise softmax of `(logits - max) / temperature`, then top-k, min-p,
   renormalize, top-p, renormalize, and draw with the request RNG.
6. Advance the grammar state, append to history, yield the token, stop on EOS.

Masks therefore apply after penalties and before temperature and every filter.
The C++ `generate` uses the same order without step 3.

## Filter semantics

* top-k (`0` disables): tokens are ranked by probability with a stable sort, so
  ties keep the lower ID; ranks beyond k are zeroed.
* min-p (`0..1`): tokens with probability below `min_p * max` are zeroed.
* top-p (`(0,1]`): walking the ranking, a token is kept while the cumulative mass
  of tokens before it is below `top_p`, so the smallest prefix reaching `top_p`
  survives.

Arguments are validated in Python before a sequence is created; invalid values
raise `ValueError` (HTTP 400).

## RNG and determinism

`iter_tokens` seeds `numpy.random.default_rng(seed)` per call and draws with
`Generator.choice`. For a fixed build, dispatch setting, prompt and parameters,
outputs repeat. Native logits are single-threaded with a fixed reduction order,
but AVX2 and scalar attention sum in different orders, so `TVARA_FORCE_SCALAR=1`
can change logits at rounding level and flip near ties. C++ `generate` uses
`std::mt19937_64` with `std::discrete_distribution`, whose algorithm is
library-defined: it does not reproduce Python draws ([native-bindings.md](native-bindings.md)).
Speculative decoding consumes the same seeded generator differently, so its
tokens differ from non-speculative sampling with the same seed.

## Grammar and JSON-schema masks

[response_constraints.py](../python/tvara/response_constraints.py) accepts only
`{"type": "json_schema", ...}` with `strict` true and compiles the schema subset
in [schema.py](../python/tvara/schema.py) into a grammar
([grammar.py](../python/tvara/grammar.py), [grammar-syntax.md](grammar-syntax.md)).
Tool requests become an `anyOf` of `{name, arguments}` objects. Grammar state is
an immutable set of symbol stacks expanded by `closure`, bounded at depth 256
and 8192 states. [token_masks.py](../python/tvara/token_masks.py) adds pending
partial UTF-8 bytes (`ByteState`), walks a byte trie of the vocabulary, prunes
subtrees on the first invalid byte, and caches masks per state in a bounded LRU
(1024 entries). Special tokens map to empty bytes and are never allowed,
except EOS, which is allowed only when the grammar accepts with no pending bytes.
An empty mask raises `ValueError`. Details: [structured-output.md](structured-output.md).

## EOS

`iter_tokens` yields the EOS token and then stops; the server takes EOS from
`tokenizer.ggml.eos_token_id`, decodes text with special tokens skipped, and
reports `finish_reason` `stop` on EOS and `length` otherwise. A constrained
response cut by `max_tokens` is incomplete JSON with `length`. Without an EOS ID
the mask never permits termination, and a grammar with no further valid
characters raises the empty-mask error.

## Speculative decoding

Implemented in Python and not exposed over HTTP. Draft and target must have
identical tokenizer fingerprints. `propose` samples up to `length` (at most 64)
draft tokens, stopping at EOS or grammar end. `verify` collects target
distributions for each prefix plus one bonus row; the native adapter uses one
`tvara_sequence_decode_batch` call when no constraint is set. Token i is
accepted with probability `min(1, p/q)`; the first rejection samples from
normalized `max(p - q, 0)`, and full acceptance samples the bonus row unless
the last token was EOS or ended the grammar. Both histories roll back to the
accepted prefix; the native rollback truncates KV and re-decodes the last
retained token. With a
grammar, cloned constraint states mask both p and q. `DraftController` adapts
length from an acceptance EMA. Native distributions are temperature-only
softmax, so penalties and filters are not part of speculative output.

Prompt lookup (`prompt_lookup`, `propose_lookup`) finds the most recent earlier
occurrence of the longest history suffix (up to 8 tokens) and proposes its
continuation as point-mass drafts, so acceptance probability equals the target
probability of each token. `verify_tree` in
[speculative_tree.py](../python/tvara/speculative_tree.py) is a greedy-only
tree verification prototype that calls the target once per accepted prefix.
Measurements and caveats: [speculative-decoding.md](speculative-decoding.md).
