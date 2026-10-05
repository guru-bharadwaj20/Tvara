import pytest
import numpy as np
from tvara.speculative import TokenizerIdentity, probabilities


class ToyModel:
    identity = TokenizerIdentity((b"a", b"b", b"c"), "fixture-v1")

    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)
        self.history = [0]

    def distribution(self):
        return self.p.copy()

    def advance(self, token):
        if not 0 <= token < len(self.p):
            raise ValueError("Invalid token")
        self.history.append(token)

    def checkpoint(self):
        return len(self.history)

    def rollback(self, checkpoint):
        if not 1 <= checkpoint <= len(self.history):
            raise ValueError("Invalid checkpoint")
        del self.history[checkpoint:]


def test_protocol_distribution_and_owned_state():
    model = ToyModel([2, 3, 5])
    np.testing.assert_allclose(probabilities(model.distribution()), [0.2, 0.3, 0.5])
    position = model.checkpoint()
    model.advance(1)
    model.rollback(position)
    assert model.history == [0]


def test_tokenizer_mismatch_is_rejected():
    from tvara.speculative import require_compatible, tokenizer_fingerprint

    draft, target = ToyModel([1, 1, 1]), ToyModel([1, 1, 1])
    require_compatible(draft, target)
    assert tokenizer_fingerprint(draft.identity) == tokenizer_fingerprint(target.identity)
    target.identity = TokenizerIdentity((b"b", b"a", b"c"), "fixture-v1")
    with pytest.raises(ValueError):
        require_compatible(draft, target)


def test_draft_proposals_preserve_each_sampling_distribution():
    from tvara.speculative import propose

    draft = ToyModel([0, 1, 0])
    proposals = propose(draft, 3, np.random.default_rng(8))
    assert [p.token for p in proposals] == [1, 1, 1]
    assert draft.history == [0, 1, 1, 1]
    for p in proposals:
        np.testing.assert_array_equal(p.distribution, [0, 1, 0])
    with pytest.raises(ValueError):
        propose(draft, 65, np.random.default_rng(0))


def test_target_verification_returns_alignment_and_bonus_distribution():
    from tvara.speculative import propose, verify

    draft, target = ToyModel([0, 1, 0]), ToyModel([0.2, 0.3, 0.5])
    proposed = propose(draft, 2, np.random.default_rng(0))
    rows = verify(target, proposed)
    assert len(rows) == 3 and target.history == [0, 1, 1]
    np.testing.assert_allclose(rows, [[0.2, 0.3, 0.5]] * 3)


def test_probability_ratio_acceptance():
    from tvara.speculative import acceptance_probability

    assert acceptance_probability([0.2, 0.3, 0.5], [0.4, 0.1, 0.5], 0) == 0.5
    assert acceptance_probability([0.2, 0.3, 0.5], [0.4, 0.1, 0.5], 1) == 1


def test_residual_samples_only_missing_target_mass():
    from tvara.speculative import residual_distribution

    np.testing.assert_allclose(residual_distribution([0.2, 0.3, 0.5], [0.4, 0.1, 0.5]), [0, 1, 0])
    with pytest.raises(ValueError):
        residual_distribution([0.2, 0.3, 0.5], [0.2, 0.3, 0.5])


def test_zero_and_extreme_probability_edges():
    from tvara.speculative import acceptance_probability, residual_distribution

    np.testing.assert_allclose(probabilities([1e308, 1e308, 0]), [0.5, 0.5, 0])
    np.testing.assert_array_equal(probabilities([0, 1e-320, 0]), [0, 1, 0])
    for bad in ([0, 0], [np.nan, 1], [np.inf, 1], [-1, 2]):
        with pytest.raises(ValueError):
            probabilities(bad)
    with pytest.raises(ValueError):
        acceptance_probability([1, 0], [0, 1], 0)
    np.testing.assert_array_equal(residual_distribution([1, 0], [0, 1]), [1, 0])


def test_fully_accepted_bonus_uses_target_distribution():
    from tvara.speculative import accepted_bonus

    assert accepted_bonus([np.array([1, 0, 0]), np.array([0, 0, 1])], np.random.default_rng(5)) == 2


def test_native_batch_and_rollback_match_uncached_conditionals():
    from pathlib import Path
    from tvara.native import NativeModel
    from tvara.speculative import NativeSpeculativeModel, Proposal, TokenizerIdentity

    root = Path.cwd()
    libraries = [
        root / "build/debug" / name
        for name in (
            "libtvara_capi.dll",
            "tvara_capi.dll",
            "libtvara_capi.so",
            "libtvara_capi.dylib",
        )
    ]
    library = next((p for p in libraries if p.exists()), None)
    model_path = root / "models/api-fixture/model.gguf"
    if library is None or not model_path.exists():
        pytest.skip("Native fixture absent; runner must build and create fixture")
    identity = TokenizerIdentity(
        tuple(bytes([i]) for i in range(256)) + (b"",), "byte-fixture-v1", 256
    )
    with (
        NativeModel(model_path, library=library) as first,
        NativeModel(model_path, library=library) as second,
    ):
        one, two = (
            NativeSpeculativeModel(first, [1, 2, 3], identity),
            NativeSpeculativeModel(second, [1, 2, 3], identity),
        )
        try:
            original = one.distribution()
            checkpoint = one.checkpoint()
            proposals = [Proposal(4, original), Proposal(5, original)]
            batch = one.verify_batch(proposals)
            expected = [two.distribution()]
            for token in [4, 5]:
                two.advance(token)
                expected.append(two.distribution())
            np.testing.assert_allclose(batch, expected, atol=1e-12)
            one.rollback(checkpoint)
            np.testing.assert_array_equal(one.distribution(), original)
            assert one.history == [1, 2, 3]
            with pytest.raises(ValueError):
                one.verify_batch([Proposal(-1, original)])
            assert one.history == [1, 2, 3]
            np.testing.assert_array_equal(one.verify_batch([])[0], original)
        finally:
            one.close()
            two.close()


def test_target_rollback_preserves_accepted_prefix():
    from tvara.speculative import rollback_target

    target = ToyModel([1, 1, 1])
    checkpoint = target.checkpoint()
    for token in [1, 2, 0]:
        target.advance(token)
    rollback_target(target, checkpoint, 1)
    assert target.history == [0, 1]
