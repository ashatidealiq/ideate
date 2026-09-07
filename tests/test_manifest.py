"""Tests for common/manifest.py (BUILD.md Phase 1 accept criteria)."""

from __future__ import annotations

import pandas as pd
import pytest

from common import config, manifest
from common.manifest import _sha256_file
from common.schemas import FileRef


@pytest.fixture(autouse=True)
def isolated_pipeline_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PIPELINE_ROOT", tmp_path)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.parquet")
    return tmp_path


def test_begin_creates_stage_dir():
    ctx = manifest.begin("01_claim", "c1", "r1")
    assert ctx.stage_dir.exists()
    assert ctx.stage_dir == config.RUNS_DIR / "c1" / "r1" / "01_claim"


def test_begin_raises_if_stage_dir_exists():
    manifest.begin("01_claim", "c1", "r1")
    with pytest.raises(FileExistsError):
        manifest.begin("01_claim", "c1", "r1")


def test_begin_allows_sibling_stages_under_same_run():
    manifest.begin("01_claim", "c1", "r1")
    ctx2 = manifest.begin("02_data", "c1", "r1")  # shares parent dirs, must not raise
    assert ctx2.stage_dir.exists()


def test_complete_writes_manifest_and_registry_row():
    ctx = manifest.begin("01_claim", "c1", "r1")
    out_path = ctx.stage_dir / "claim.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(out_path)
    out_ref = FileRef(path="claim.parquet", sha256=_sha256_file(out_path))

    m = manifest.complete(ctx, outputs=[out_ref], code_commit="abc123", common_version="0.0.1")

    assert m.status == "complete"
    assert (ctx.stage_dir / "manifest.json").exists()
    registry = pd.read_parquet(config.REGISTRY_PATH)
    assert len(registry) == 1
    assert registry.iloc[0]["stage"] == "01_claim"
    assert registry.iloc[0]["status"] == "complete"


def test_read_input_returns_data_on_hash_match():
    ctx = manifest.begin("01_claim", "c1", "r1")
    out_path = ctx.stage_dir / "claim.parquet"
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_parquet(out_path)
    out_ref = FileRef(path="claim.parquet", sha256=_sha256_file(out_path))
    manifest.complete(ctx, outputs=[out_ref])

    read_back = manifest.read_input("c1", "r1", "01_claim", "claim.parquet")
    pd.testing.assert_frame_equal(read_back, df)


def test_read_input_raises_on_tampered_file():
    ctx = manifest.begin("01_claim", "c1", "r1")
    out_path = ctx.stage_dir / "claim.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(out_path)
    out_ref = FileRef(path="claim.parquet", sha256=_sha256_file(out_path))
    manifest.complete(ctx, outputs=[out_ref])

    # Tamper with the file after its hash was recorded in the manifest.
    pd.DataFrame({"a": [9, 9, 9]}).to_parquet(out_path)
    tampered_sha256 = _sha256_file(out_path)

    with pytest.raises(ValueError) as excinfo:
        manifest.read_input("c1", "r1", "01_claim", "claim.parquet")

    message = str(excinfo.value)
    assert str(out_path) in message
    assert out_ref.sha256 in message
    assert tampered_sha256 in message


def test_read_input_raises_on_unrecorded_filename():
    ctx = manifest.begin("01_claim", "c1", "r1")
    manifest.complete(ctx, outputs=[])
    with pytest.raises(KeyError):
        manifest.read_input("c1", "r1", "01_claim", "never_written.parquet")


def test_registry_append_is_idempotent_per_claim_run_stage():
    ctx = manifest.begin("01_claim", "c1", "r1")
    manifest.complete(ctx, outputs=[])
    # Simulate a retry (e.g. after a crash) re-completing the same stage.
    manifest.complete(ctx, outputs=[])

    registry = pd.read_parquet(config.REGISTRY_PATH)
    assert len(registry) == 1


def test_registry_append_adds_a_row_per_distinct_stage():
    ctx1 = manifest.begin("01_claim", "c1", "r1")
    manifest.complete(ctx1, outputs=[])
    ctx2 = manifest.begin("02_data", "c1", "r1")
    manifest.complete(ctx2, outputs=[])

    registry = pd.read_parquet(config.REGISTRY_PATH)
    assert len(registry) == 2
    assert set(registry["stage"]) == {"01_claim", "02_data"}


def test_halt_and_fail_write_terminal_status():
    ctx = manifest.begin("01_claim", "c1", "r1")
    m = manifest.halt(ctx, halt_reason="tradable == False")
    assert m.status == "halted"
    assert m.halt_reason == "tradable == False"

    ctx2 = manifest.begin("01_claim", "c2", "r1")
    m2 = manifest.fail(ctx2, reason="unexpected KeyError")
    assert m2.status == "failed"
    assert m2.halt_reason == "unexpected KeyError"
