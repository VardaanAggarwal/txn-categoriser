import pytest

from txncat.llm.base import LLMError, TransientLLMError
from txncat.retry import with_retries


class Flaky:
    def __init__(self, failures: list[Exception]):
        self.failures = failures
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


def test_retries_transient_errors_then_succeeds():
    fn, sleeps = Flaky([TransientLLMError("429"), TransientLLMError("503")]), []
    assert with_retries(fn, sleep=sleeps.append, rand=lambda: 0.0) == "ok"
    assert fn.calls == 3
    assert sleeps == [0.25, 0.5]  # half of 0.5s and 1s with zero jitter


def test_backoff_doubles_and_is_capped():
    fn, sleeps = Flaky([TransientLLMError("x")] * 5), []
    with_retries(fn, attempts=6, base_delay=1.0, max_delay=4.0, sleep=sleeps.append, rand=lambda: 1.0)
    assert sleeps == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_gives_up_after_the_last_attempt():
    fn = Flaky([TransientLLMError("x")] * 3)
    with pytest.raises(TransientLLMError):
        with_retries(fn, attempts=3, sleep=lambda _: None)
    assert fn.calls == 3


def test_permanent_errors_are_not_retried():
    fn = Flaky([LLMError("400 bad request")])
    with pytest.raises(LLMError):
        with_retries(fn, sleep=lambda _: None)
    assert fn.calls == 1


def test_rejects_zero_attempts():
    with pytest.raises(ValueError):
        with_retries(lambda: 1, attempts=0)
