"""Validate the roadmap's stable IDs, status totals, and evidence links."""

from pathlib import Path
import re


def audit(root: Path) -> tuple[int, int]:
    text = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    rows = re.findall(r"^- ([\u2705\u274c]) T(\d{3}) (.+)$", text, re.M)
    if [int(n) for _, n, _ in rows] != list(range(1, 271)):
        raise ValueError("Expected exactly T001 through T270 in order")
    if len(re.findall(r"^### Phase", text, re.M)) != 18:
        raise ValueError("Expected 18 roadmap phases")
    for _, number, detail in rows:
        for target in re.findall(r"\]\(([^)]+)\)", detail):
            if not target.startswith("https://") and not (root / target.split("#")[0]).exists():
                raise ValueError(f"T{number}: missing evidence file {target}")
    done = sum(mark == chr(0x2705) for mark, _, _ in rows)
    if f"{done} tasks are completed and {270 - done} are not completed" not in text:
        raise ValueError("Task totals do not match markers")
    return done, 270 - done


if __name__ == "__main__":
    done, pending = audit(Path(__file__).resolve().parents[1])
    print(f"270 tasks in 18 phases: {done} complete; {pending} pending")
