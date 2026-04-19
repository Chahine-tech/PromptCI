from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from pytest_prompts.config import settings
from pytest_prompts.snapshot import Snapshot, SnapshotStore

app = typer.Typer(
    help="pytest for LLM prompts — tests, regressions, CI.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def run(
    path: str = typer.Argument(".", help="Test path (file or directory)."),
    snapshot_dir: str = typer.Option(
        settings.snapshot_dir, "--snapshot-dir", help="Where to write snapshots."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run prompt tests via pytest and summarize results."""
    store = SnapshotStore(snapshot_dir)
    # Fresh run → clear previous snapshots so --summary reflects this run only.
    _clear_dir(store.root)

    args = [
        sys.executable,
        "-m",
        "pytest",
        path,
        f"--pytest-prompts-snapshot-dir={snapshot_dir}",
    ]
    if verbose:
        args.append("-v")
    else:
        args.append("-q")

    result = subprocess.run(args, check=False)  # noqa: S603

    snapshots = store.all()
    _print_summary(snapshots)

    raise typer.Exit(code=result.returncode)


@app.command()
def diff(
    base: str = typer.Argument(..., help="Path to baseline snapshot directory."),
    head: str = typer.Argument(..., help="Path to current snapshot directory."),
    threshold: float = typer.Option(
        0.05, "--threshold", help="Regression threshold (fraction, default 0.05)."
    ),
) -> None:
    """Compare two snapshot directories and report regressions."""
    base_store = SnapshotStore(base)
    head_store = SnapshotStore(head)

    base_map = {s.test_id: s for s in base_store.all()}
    head_map = {s.test_id: s for s in head_store.all()}

    if not base_map:
        console.print(f"[red]No snapshots in baseline:[/red] {base}")
        raise typer.Exit(code=2)
    if not head_map:
        console.print(f"[red]No snapshots in head:[/red] {head}")
        raise typer.Exit(code=2)

    regressions = _compute_regressions(base_map, head_map, threshold)
    _print_diff(base_map, head_map, regressions)

    raise typer.Exit(code=1 if regressions else 0)


def _clear_dir(path: Path) -> None:
    if not path.is_dir():
        return
    for f in path.glob("*.json"):
        f.unlink()


def _print_summary(snapshots: list[Snapshot]) -> None:
    if not snapshots:
        console.print("\n[yellow]No pytest-prompts snapshots recorded.[/yellow]")
        return

    passed = sum(1 for s in snapshots if s.passed)
    failed = len(snapshots) - passed
    total_tokens = sum(s.input_tokens + s.output_tokens for s in snapshots)
    total_cost = sum(s.cost_usd for s in snapshots)

    table = Table(title="pytest-prompts results", show_lines=False)
    table.add_column("Test", style="cyan", overflow="fold")
    table.add_column("Model", style="magenta")
    table.add_column("Tokens", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Status", justify="center")

    for s in snapshots:
        status = Text("PASS", style="green") if s.passed else Text("FAIL", style="red")
        table.add_row(
            s.test_id,
            s.model,
            str(s.input_tokens + s.output_tokens),
            f"{s.latency_ms}ms",
            status,
        )

    console.print()
    console.print(table)
    summary = (
        f"[bold]{passed}[/bold] passed, "
        f"[bold]{failed}[/bold] failed — "
        f"{total_tokens} tokens total — ${total_cost:.4f}"
    )
    console.print(summary)


def _compute_regressions(
    base: dict[str, Snapshot],
    head: dict[str, Snapshot],
    threshold: float,
) -> list[tuple[str, str]]:
    """Return list of (test_id, reason) for each regression."""
    regressions: list[tuple[str, str]] = []
    for test_id, h in head.items():
        b = base.get(test_id)
        if b is None:
            continue
        if b.passed and not h.passed:
            regressions.append((test_id, "pass → fail"))
            continue
        base_tokens = b.input_tokens + b.output_tokens
        head_tokens = h.input_tokens + h.output_tokens
        if base_tokens > 0 and (head_tokens - base_tokens) / base_tokens > threshold:
            pct = (head_tokens - base_tokens) / base_tokens * 100
            regressions.append(
                (test_id, f"tokens {base_tokens} → {head_tokens} (+{pct:.0f}%)")
            )
            continue
        if b.latency_ms > 0 and (h.latency_ms - b.latency_ms) / b.latency_ms > threshold:
            pct = (h.latency_ms - b.latency_ms) / b.latency_ms * 100
            regressions.append(
                (test_id, f"latency {b.latency_ms}ms → {h.latency_ms}ms (+{pct:.0f}%)")
            )
    return regressions


def _print_diff(
    base: dict[str, Snapshot],
    head: dict[str, Snapshot],
    regressions: list[tuple[str, str]],
) -> None:
    table = Table(title="pytest-prompts diff", show_lines=False)
    table.add_column("Test", style="cyan", overflow="fold")
    table.add_column("Base", justify="right")
    table.add_column("Head", justify="right")
    table.add_column("Status", justify="center")

    reg_ids = {tid for tid, _ in regressions}

    for test_id in sorted(set(base) | set(head)):
        b = base.get(test_id)
        h = head.get(test_id)
        base_cell = _cell(b)
        head_cell = _cell(h)
        if test_id in reg_ids:
            status = Text("REGRESSION", style="red bold")
        elif b is None:
            status = Text("new", style="blue")
        elif h is None:
            status = Text("removed", style="yellow")
        else:
            status = Text("ok", style="green")
        table.add_row(test_id, base_cell, head_cell, status)

    console.print()
    console.print(table)

    if regressions:
        console.print("\n[red bold]Regressions:[/red bold]")
        for tid, reason in regressions:
            console.print(f"  • {tid} — {reason}")
    else:
        console.print("\n[green]No regressions detected.[/green]")


def _cell(s: Snapshot | None) -> str:
    if s is None:
        return "-"
    status = "✓" if s.passed else "✗"
    tokens = s.input_tokens + s.output_tokens
    return f"{status} {tokens}t {s.latency_ms}ms"


if __name__ == "__main__":
    app()
