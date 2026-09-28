# Performance measurement protocol

No benchmark results have been collected yet. Report prefill and decode separately.
Use identical model bytes, thread counts, input token counts, generated token counts,
and sampling settings when comparing Tvara with llama.cpp.

| Metric | Measurement |
|---|---|
| Prefill tokens/s | Prompt token count divided by prefill wall time |
| Decode tokens/s | Generated decode tokens divided by decode wall time |
| TTFT | Request submission to first emitted token |
| Inter-token latency | Gaps between emitted tokens, p50 and p99 |
| Aggregate throughput | Total completed tokens divided by load-test duration |
| Memory use | Peak resident memory plus configured KV capacity |
| Prefix reuse | Reused prompt tokens, hit rate, and saved prefill time |
| Speculation | Accepted/proposed tokens and matched-baseline speed ratio |
| Grammar cost | Mask computation time and schema compliance rate |

Record CPU, RAM, OS, compiler flags, library versions, power settings, warm-up count,
repetitions, and raw timings. Report median and spread, not the fastest run alone.
Measure STREAM on the same hardware before reporting achieved bandwidth fractions.
Keep synthetic load seeds and conversation-length distributions reproducible.
