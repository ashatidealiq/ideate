"""Write, read, and verify stage manifests (DESIGN.md §3.2-§3.4).

`begin(stage, claim_id, run_id)` creates the stage's output directory and
fails if it already exists — stage directories are write-once (CLAUDE.md);
a re-run is a new `run_id` with `parent_run_id` set, never an overwrite.
It returns a `StageContext` that `complete()` / `halt()` / `fail()` turn
into a full `schemas.Manifest`, written as `manifest.json` in that
directory, with `status` one of `complete`, `halted`, `failed` — there is
no fourth outcome.

`read_input(claim_id, run_id, stage, filename)` is the only way a stage may
read a prior stage's output. It recomputes the file's sha256 and compares
it against the hash recorded in that prior stage's manifest; a mismatch
raises, naming the path and both hashes.

Every `complete()` / `halt()` / `fail()` also appends one row to
`registry.parquet` (DESIGN §3.4). The append is atomic (written to a temp
file, then rename-replaced) and idempotent per `(claim_id, run_id, stage)`:
appending the same key twice is a no-op the second time.
"""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import config
from common.schemas import FileRef, LLMCallInfo, Manifest

REGISTRY_COLUMNS = [
    "claim_id", "run_id", "parent_run_id", "stage",
    "status", "halt_reason", "completed_utc", "trial_count",
]


@dataclass
class StageContext:
    """State captured by `begin()` and consumed by `complete()`/`halt()`/`fail()`."""

    stage: str
    claim_id: str
    run_id: str
    parent_run_id: str | None
    stage_dir: Path
    started_utc: datetime


def stage_dir(claim_id: str, run_id: str, stage: str) -> Path:
    return config.RUNS_DIR / claim_id / run_id / stage


def begin(stage: str, claim_id: str, run_id: str, parent_run_id: str | None = None) -> StageContext:
    """Creates the stage directory. Raises `FileExistsError` if it already
    exists."""
    d = stage_dir(claim_id, run_id, stage)
    d.mkdir(parents=True, exist_ok=False)
    return StageContext(
        stage=stage,
        claim_id=claim_id,
        run_id=run_id,
        parent_run_id=parent_run_id,
        stage_dir=d,
        started_utc=datetime.now(timezone.utc),
    )


def _write_manifest(ctx: StageContext, manifest: Manifest) -> None:
    (ctx.stage_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def _finish(
    ctx: StageContext,
    status: str,
    *,
    inputs: list[FileRef] | None = None,
    outputs: list[FileRef] | None = None,
    halt_reason: str | None = None,
    code_commit: str = "",
    common_version: str = "",
    catalogue_sha256: str | None = None,
    llm: LLMCallInfo | None = None,
    trial_count: int = 0,
) -> Manifest:
    manifest = Manifest(
        stage=ctx.stage,
        claim_id=ctx.claim_id,
        run_id=ctx.run_id,
        parent_run_id=ctx.parent_run_id,
        started_utc=ctx.started_utc,
        completed_utc=datetime.now(timezone.utc),
        status=status,
        halt_reason=halt_reason,
        inputs=inputs or [],
        outputs=outputs or [],
        code_commit=code_commit,
        common_version=common_version,
        catalogue_sha256=catalogue_sha256,
        llm=llm,
    )
    _write_manifest(ctx, manifest)
    _registry_append(manifest, trial_count=trial_count)
    return manifest


def complete(
    ctx: StageContext,
    outputs: list[FileRef],
    inputs: list[FileRef] | None = None,
    code_commit: str = "",
    common_version: str = "",
    catalogue_sha256: str | None = None,
    llm: LLMCallInfo | None = None,
    trial_count: int = 0,
) -> Manifest:
    """Writes a `status: complete` manifest and appends the registry row."""
    return _finish(
        ctx, "complete", inputs=inputs, outputs=outputs,
        code_commit=code_commit, common_version=common_version,
        catalogue_sha256=catalogue_sha256, llm=llm, trial_count=trial_count,
    )


def halt(
    ctx: StageContext,
    halt_reason: str,
    inputs: list[FileRef] | None = None,
    code_commit: str = "",
    common_version: str = "",
    catalogue_sha256: str | None = None,
    llm: LLMCallInfo | None = None,
    trial_count: int = 0,
) -> Manifest:
    """Writes a `status: halted` manifest: the stage could not proceed and
    said why, rather than crashing (CLAUDE.md "Halting"). `llm` records
    provenance for a real LLM call that happened before the halt (e.g. a
    tradability verdict of False still made a real call)."""
    return _finish(
        ctx, "halted", inputs=inputs, halt_reason=halt_reason,
        code_commit=code_commit, common_version=common_version,
        catalogue_sha256=catalogue_sha256, llm=llm, trial_count=trial_count,
    )


def fail(
    ctx: StageContext,
    reason: str,
    inputs: list[FileRef] | None = None,
    code_commit: str = "",
    common_version: str = "",
    trial_count: int = 0,
) -> Manifest:
    """Writes a `status: failed` manifest: the stage crashed. Callers should
    wrap stage bodies so an unexpected exception still reaches this rather
    than leaving the stage directory without a manifest."""
    return _finish(
        ctx, "failed", inputs=inputs, halt_reason=reason,
        code_commit=code_commit, common_version=common_version,
        trial_count=trial_count,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_input(claim_id: str, run_id: str, stage: str, filename: str) -> pd.DataFrame:
    """Reads one file from a prior stage's output, verifying it against the
    sha256 recorded in that stage's manifest. Raises on any hash mismatch,
    naming the path and both hashes."""
    prior_dir = stage_dir(claim_id, run_id, stage)
    manifest = Manifest.model_validate_json((prior_dir / "manifest.json").read_text())

    matches = [o for o in manifest.outputs if o.path == filename or Path(o.path).name == filename]
    if not matches:
        raise KeyError(f"{filename!r} is not a recorded output of stage {stage!r} for run {run_id!r}")
    expected_sha256 = matches[0].sha256

    file_path = prior_dir / filename
    actual_sha256 = _sha256_file(file_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"hash mismatch reading {file_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return pd.read_parquet(file_path)


def get_trial_count(claim_id: str) -> int:
    """The cumulative number of backtests executed under `claim_id`, across
    every run and stage recorded in the registry (DESIGN §3.4) -- the
    latest (max) `trial_count` value for that claim, or 0 if the claim has
    no registry rows yet. `gates.py`'s G4 (deflated Sharpe) calls this
    directly rather than taking `trial_count` as an argument, so it can
    never be passed a stale or spoofed value."""
    path = config.REGISTRY_PATH
    if not path.exists():
        return 0
    registry = pd.read_parquet(path)
    rows = registry[registry["claim_id"] == claim_id]
    if rows.empty:
        return 0
    return int(rows["trial_count"].max())


@contextmanager
def _registry_lock(lock_path: Path, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for registry lock {lock_path}")
            time.sleep(0.02)
    try:
        yield
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)


def _registry_append(manifest: Manifest, trial_count: int) -> None:
    row = {
        "claim_id": manifest.claim_id,
        "run_id": manifest.run_id,
        "parent_run_id": manifest.parent_run_id,
        "stage": manifest.stage,
        "status": manifest.status,
        "halt_reason": manifest.halt_reason,
        "completed_utc": manifest.completed_utc,
        "trial_count": trial_count,
    }
    path = config.REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")

    with _registry_lock(lock_path):
        if path.exists():
            existing = pd.read_parquet(path)
        else:
            existing = pd.DataFrame(columns=REGISTRY_COLUMNS)

        key = (row["claim_id"], row["run_id"], row["stage"])
        if len(existing) and (
            (existing["claim_id"] == key[0])
            & (existing["run_id"] == key[1])
            & (existing["stage"] == key[2])
        ).any():
            return  # idempotent: this (claim_id, run_id, stage) is already recorded

        new_row = pd.DataFrame([row])
        updated = new_row if existing.empty else pd.concat([existing, new_row], ignore_index=True)
        tmp_path = path.with_name(path.name + ".tmp")
        updated.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)  # atomic on POSIX and Windows
