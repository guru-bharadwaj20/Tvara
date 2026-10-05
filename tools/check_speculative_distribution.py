import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from tvara.speculative import (
    acceptance_probability,
    residual_distribution,
    speculative_step,
    TokenizerIdentity,
)

rng = np.random.default_rng(182736)
records = []
for p, q in [
    (np.array([0.2, 0.3, 0.5]), np.array([0.6, 0.3, 0.1])),
    (np.array([0.0, 0.7, 0.3]), np.array([0.5, 0.5, 0.0])),
    (np.array([0.1, 0.2, 0.7]), np.array([0.1, 0.2, 0.7])),
]:
    draws = 200000
    proposals = rng.choice(3, size=draws, p=q)
    ratios = np.array([acceptance_probability(p, q, i) if q[i] else 0 for i in range(3)])
    accepted = rng.random(draws) < ratios[proposals]
    samples = proposals.copy()
    if not np.all(accepted):
        samples[~accepted] = rng.choice(
            3, size=int((~accepted).sum()), p=residual_distribution(p, q)
        )
    observed = np.bincount(samples, minlength=3) / draws
    bound = np.sqrt(np.log(2 * 18 / 0.001) / (2 * draws))
    assert np.max(np.abs(observed - p)) < bound
    analytical = q * ratios
    rejection = 1 - analytical.sum()
    if rejection > 1e-14:
        analytical += rejection * residual_distribution(p, q)
    np.testing.assert_allclose(analytical, p, atol=1e-14)
    records.append(
        {
            "target": p.tolist(),
            "draft": q.tolist(),
            "observed": observed.tolist(),
            "draws": draws,
            "simultaneous_hoeffding_bound": float(bound),
            "maximum_error": float(np.max(np.abs(observed - p))),
        }
    )


class DriverModel:
    identity = TokenizerIdentity((b"a", b"b", b"c"), "statistical-fixture-v1")

    def __init__(self, p):
        self.p = p
        self.history = [0]

    def distribution(self):
        return self.p.copy()

    def advance(self, token):
        self.history.append(token)

    def checkpoint(self):
        return len(self.history)

    def rollback(self, checkpoint):
        del self.history[checkpoint:]


for record in records:
    p = np.asarray(record["target"])
    q = np.asarray(record["draft"])
    driver_draws = 10000
    counts = np.zeros(3, dtype=int)
    for _ in range(driver_draws):
        step = speculative_step(DriverModel(q), DriverModel(p), 2, rng)
        counts[step.tokens[0]] += 1
    driver_observed = counts / driver_draws
    driver_bound = np.sqrt(np.log(2 * 18 / 0.001) / (2 * driver_draws))
    assert np.max(np.abs(driver_observed - p)) < driver_bound
    record.update(
        driver_draws=driver_draws,
        driver_observed=driver_observed.tolist(),
        driver_hoeffding_bound=float(driver_bound),
    )

result = {
    "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed": 182736,
    "familywise_failure_probability": 0.001,
    "scope": "independent one-token acceptance/residual distribution, eighteen simultaneous coordinate checks, including actual driver emission; not trained-model quality",
    "cases": records,
}
Path("bench/results").mkdir(parents=True, exist_ok=True)
Path("bench/results/speculative-distribution.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
