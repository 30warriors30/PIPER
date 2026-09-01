from pathlib import Path
import pytest
from utils.io import RunStore, append_jsonl


def test_resume_skips_only_completed(tmp_path: Path) -> None:
    store = RunStore.initialize(tmp_path / "run", {"schema_version":1,"name":"test"})
    append_jsonl(store.samples_path, {"sample_id":"ok","status":"completed"})
    append_jsonl(store.samples_path, {"sample_id":"retry","status":"error"})
    assert store.completed_sample_ids() == {"ok"}


def test_resume_rejects_mismatch(tmp_path: Path) -> None:
    RunStore.initialize(tmp_path / "run", {"schema_version":1,"name":"first"})
    with pytest.raises(ValueError, match="mismatch"):
        RunStore.initialize(tmp_path / "run", {"schema_version":1,"name":"second"}, resume=True)
