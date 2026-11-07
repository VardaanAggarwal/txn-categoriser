"""A scripted LLM for tests: no network, fully deterministic."""

from collections.abc import Callable


class FakeLLM:
    name = "fake"

    def __init__(self, responses: list[str | Exception] | Callable[[str, str], str]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if callable(self._responses):
            return self._responses(system, user)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
