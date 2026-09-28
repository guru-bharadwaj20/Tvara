"""Greedy baseline using complete-prefix recomputation."""


def greedy_generate(model, tokens, max_new_tokens, eos_token_id=None):
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    sequence = list(tokens)
    if not sequence:
        raise ValueError("Generation requires a nonempty prompt")
    generated = []
    for _ in range(max_new_tokens):
        token = int(model.forward(sequence)[-1].argmax())
        sequence.append(token)
        generated.append(token)
        if token == eos_token_id:
            break
    return generated
