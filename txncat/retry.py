"""Retry with exponential backoff and jitter, for transient LLM failures only."""

import random
import time
from collections.abc import Callable
from typing import TypeVar

from .llm.base import TransientLLMError

T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Call fn, retrying TransientLLMError up to `attempts` times in total.

    Delays double each time (0.5s, 1s, 2s...) up to max_delay. "Equal jitter" keeps at least half
    of each delay and randomises the rest, so many clients hitting a rate limit at once don't all
    retry at the same moment. Any other exception is raised immediately.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return fn()
        except TransientLLMError:
            if attempt == attempts - 1:
                raise
            delay = min(max_delay, base_delay * 2**attempt)
            sleep(delay / 2 + rand() * delay / 2)
    raise AssertionError("unreachable")
