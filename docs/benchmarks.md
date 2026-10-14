# Performance measurement protocol

Recorded results are tabulated in [performance.md](performance.md), generated from
`bench/results/` by `tools/render_performance_tables.py`. Report prefill and decode separately.
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

## Final measurement conditions

Every final benchmark records the actual UTC execution time. Commit author and committer dates are historical metadata and do not describe test execution time. Earlier prefill artifacts without timestamps retain that limitation; dates are not reconstructed. Repeated native timings report raw samples, range and median. CPU temperature and frequency are unavailable to this process, cooldown is not imposed and concurrent work is uncontrolled, so these measurements do not establish thermal stability or an idle-machine performance baseline.
