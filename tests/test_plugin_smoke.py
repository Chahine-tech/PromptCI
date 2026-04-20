"""Smoke test: verify plugin fixture + snapshot writing works end-to-end.

Uses pytester to run an inner pytest session against MockRunner so we don't
hit the real Anthropic API.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


def test_fixture_injects_runner_and_writes_snapshot(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_inner="""
        from pytest_prompts import prompt_test
        from pytest_prompts.plugin import _RecordingRunner
        from pytest_prompts.runner import MockRunner

        @prompt_test(model="mock-model")
        def test_uses_runner(runner, monkeypatch):
            # Swap the inner Runner for MockRunner so we don't hit the network.
            runner._inner = MockRunner(canned_output="hello world")
            result = runner.run(prompt="Say hi")
            assert result.output == "hello world"
            assert result.tokens_used == 15
        """
    )
    snap_dir = pytester.path / "snaps"
    result = pytester.runpytest("-q", f"--pytest-prompts-snapshot-dir={snap_dir}")
    result.assert_outcomes(passed=1)

    snapshots = list(snap_dir.glob("*.json"))
    assert len(snapshots) == 1, f"expected 1 snapshot, got {snapshots}"

    data = json.loads(snapshots[0].read_text())
    assert data["passed"] is True
    assert data["output"] == "hello world"
    assert data["model"] == "mock"  # MockRunner default
    assert "test_uses_runner" in data["test_id"]


def test_no_snapshot_when_not_decorated(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_inner="""
        from pytest_prompts.plugin import _RecordingRunner
        from pytest_prompts.runner import MockRunner

        def test_no_decorator(runner):
            runner._inner = MockRunner()
            runner.run(prompt="anything")
        """
    )
    snap_dir = pytester.path / "snaps"
    result = pytester.runpytest("-q", f"--pytest-prompts-snapshot-dir={snap_dir}")
    result.assert_outcomes(passed=1)

    assert not snap_dir.exists() or list(snap_dir.glob("*.json")) == []


def test_failed_test_snapshot_captures_error(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_inner="""
        from pytest_prompts import prompt_test
        from pytest_prompts.runner import MockRunner

        @prompt_test()
        def test_will_fail(runner):
            runner._inner = MockRunner(canned_output="nope")
            result = runner.run(prompt="x")
            assert result.output == "expected"
        """
    )
    snap_dir = pytester.path / "snaps"
    result = pytester.runpytest("-q", f"--pytest-prompts-snapshot-dir={snap_dir}")
    result.assert_outcomes(failed=1)

    snapshots = list(Path(snap_dir).glob("*.json"))
    assert len(snapshots) == 1
    data = json.loads(snapshots[0].read_text())
    assert data["passed"] is False
    assert data["error"] is not None


def test_judge_verdict_recorded_in_snapshot(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_inner="""
        from pytest_prompts import prompt_test
        from pytest_prompts.runner import MockRunner

        @prompt_test()
        def test_with_judge(runner):
            runner._inner = MockRunner(
                canned_output="Paris is the capital of France", canned_verdict=True
            )
            result = runner.run(prompt="What is the capital of France?")
            verdict = runner.judge(result, "The answer mentions Paris")
            assert verdict.verdict is True
        """
    )
    snap_dir = pytester.path / "snaps"
    result = pytester.runpytest("-q", f"--pytest-prompts-snapshot-dir={snap_dir}")
    result.assert_outcomes(passed=1)

    snapshots = list(snap_dir.glob("*.json"))
    assert len(snapshots) == 1
    data = json.loads(snapshots[0].read_text())
    assert data["judge_calls"] == [
        {
            "verdict": True,
            "reasoning": "mock judge",
            "criterion": "The answer mentions Paris",
            "input_tokens": 5,
            "output_tokens": 3,
            "cost_usd": 0.0,
        }
    ]


def test_judge_failure_fails_test(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_inner="""
        from pytest_prompts import prompt_test
        from pytest_prompts.runner import MockRunner

        @prompt_test()
        def test_judge_says_no(runner):
            runner._inner = MockRunner(canned_output="I don't know", canned_verdict=False)
            result = runner.run(prompt="x")
            verdict = runner.judge(result, "The answer mentions Paris")
            assert verdict.verdict, verdict.reasoning
        """
    )
    snap_dir = pytester.path / "snaps"
    result = pytester.runpytest("-q", f"--pytest-prompts-snapshot-dir={snap_dir}")
    result.assert_outcomes(failed=1)

    data = json.loads(list(snap_dir.glob("*.json"))[0].read_text())
    assert data["passed"] is False
    assert data["judge_calls"][0]["verdict"] is False
