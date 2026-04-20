from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pytest_prompts.runner import JudgeResult, RunResult

SAFE_ID = re.compile(r"[^a-zA-Z0-9_.-]")


@dataclass(slots=True)
class Snapshot:
    test_id: str
    passed: bool
    model: str
    prompt_hash: str
    output: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    timestamp: float
    error: str | None = None
    judge_calls: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_result(
        cls,
        test_id: str,
        passed: bool,
        result: RunResult,
        error: str | None = None,
        judge_calls: list[JudgeResult] | None = None,
    ) -> Snapshot:
        return cls(
            test_id=test_id,
            passed=passed,
            model=result.model,
            prompt_hash=result.prompt_hash,
            output=result.output,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            timestamp=time.time(),
            error=error,
            judge_calls=[asdict(j) for j in (judge_calls or [])],
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _safe_filename(test_id: str) -> str:
    return SAFE_ID.sub("_", test_id) + ".json"


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, test_id: str) -> Path:
        return self.root / _safe_filename(test_id)

    def write(self, snapshot: Snapshot) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(snapshot.test_id)
        path.write_text(snapshot.to_json(), encoding="utf-8")
        return path

    def read(self, test_id: str) -> Snapshot | None:
        path = self.path_for(test_id)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Snapshot(**data)

    def all(self) -> list[Snapshot]:
        if not self.root.is_dir():
            return []
        out: list[Snapshot] = []
        for path in sorted(self.root.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append(Snapshot(**data))
        return out
