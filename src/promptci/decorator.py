from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

F = TypeVar("F", bound=Callable[..., object])

PROMPT_TEST_ATTR = "__promptci_meta__"


@dataclass(slots=True, frozen=True)
class PromptTestMeta:
    model: str | None
    timeout: float | None
    max_tokens: int | None


def prompt_test(
    model: str | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
) -> Callable[[F], F]:
    """Mark a pytest test as a PromptCI prompt test.

    The test must declare a `runner` parameter, which will be injected
    by the pytest plugin with a Runner configured for this test.
    """

    def decorator(func: F) -> F:
        setattr(
            func,
            PROMPT_TEST_ATTR,
            PromptTestMeta(model=model, timeout=timeout, max_tokens=max_tokens),
        )
        return func

    return decorator


def get_meta(func: Callable[..., object]) -> PromptTestMeta | None:
    return getattr(func, PROMPT_TEST_ATTR, None)
