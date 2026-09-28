import json
from tvara.reference.oracle import load_oracle, oracle_logits

model = load_oracle("models/tiny-llama", "tests/fixtures/reference-model.json")
prompt = [1, 2, 3]
ids = prompt.copy()
output = []
for _ in range(6):
    token = int(oracle_logits(model, ids)[-1].argmax())
    ids.append(token)
    output.append(token)
with open("tests/fixtures/greedy.json", "w", encoding="utf-8") as f:
    json.dump(
        {"prompt": prompt, "generated": output, "algorithm": "greedy float32 full prefix"},
        f,
        indent=2,
    )
    f.write("\n")
