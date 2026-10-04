"""Single-owner iteration scheduler; backend adapters own numerical execution."""

from dataclasses import dataclass, field
from enum import Enum
import time


class State(Enum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Request:
    identifier: str
    prompt: tuple[int, ...]
    max_tokens: int
    priority: int = 0
    state: State = State.QUEUED
    position: int = 0
    output: list[int] = field(default_factory=list)
    submitted: float = field(default_factory=time.perf_counter)
    token_times: list[float] = field(default_factory=list)
    error: str | None = None

    def __post_init__(self):
        self.prompt = tuple(self.prompt)
        if type(self.priority) is not int or not -100 <= self.priority <= 100:
            raise ValueError("Invalid priority")
        if (
            not self.identifier
            or not self.prompt
            or any(type(t) is not int or t < 0 for t in self.prompt)
            or type(self.max_tokens) is not int
            or self.max_tokens < 0
        ):
            raise ValueError("Invalid scheduler request")
