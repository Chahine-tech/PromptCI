from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from promptci.config import settings
from promptci.decorator import get_meta
from promptci.runner import Runner, RunResult
from promptci.snapshot import Snapshot, SnapshotStore

_LAST_RESULT_KEY: pytest.StashKey[RunResult] = pytest.StashKey()


class _RecordingRunner:
    """Wraps a Runner and stores the last RunResult on the pytest item."""

    def __init__(self, inner: Runner, item: pytest.Item) -> None:
        self._inner = inner
        self._item = item

    def run(
        self,
        prompt: str | Path,
        input: str | None = None,
        variables: dict[str, str] | None = None,
        system: str | None = None,
    ) -> RunResult:
        result = self._inner.run(
            prompt=prompt, input=input, variables=variables, system=system
        )
        self._item.stash[_LAST_RESULT_KEY] = result
        return result


@pytest.fixture
def runner(request: pytest.FixtureRequest) -> _RecordingRunner:
    func: Callable[..., object] | None = getattr(request, "function", None)
    meta = get_meta(func) if func is not None else None
    if meta is None:
        inner = Runner()
    else:
        inner = Runner(
            model=meta.model,
            timeout=meta.timeout,
            max_tokens=meta.max_tokens,
        )
    return _RecordingRunner(inner, request.node)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield

    if report.when != "call":
        return report

    func: Callable[..., object] | None = getattr(item, "function", None)
    if func is None or get_meta(func) is None:
        return report

    result = item.stash.get(_LAST_RESULT_KEY, None)
    if result is None:
        return report

    error = None if report.passed else (report.longreprtext or "failed")
    snapshot = Snapshot.from_result(
        test_id=item.nodeid,
        passed=report.passed,
        result=result,
        error=error,
    )

    root = (
        item.config.getoption("--promptci-snapshot-dir", default=None)
        or settings.snapshot_dir
    )
    SnapshotStore(root).write(snapshot)
    return report


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--promptci-snapshot-dir",
        action="store",
        default=None,
        help="Directory to write PromptCI snapshots (overrides settings).",
    )
