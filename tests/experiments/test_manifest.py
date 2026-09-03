import json
from pathlib import Path

from experiments.config import ParetoExperimentConfig
from experiments.manifest import build_manifest
from utils.io import iter_jsonl


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        ids = list(range(len(str(text).split())))
        if add_special_tokens:
            ids = [99] + ids
        return {"input_ids": ids}

    def build_inputs_with_special_tokens(self, ids):
        return [99] + list(ids)

    def decode(self, ids, skip_special_tokens=True):
        values = [int(value) for value in ids if not (skip_special_tokens and int(value) == 99)]
        return " ".join(f"t{value}" for value in values)


def test_build_manifest_accepts_processed_c4_prompt_and_natural_text(tmp_path: Path) -> None:
    dataset_path = tmp_path / "processed_c4.json"
    dataset_path.write_text(
        "".join(
            json.dumps({"prompt": f"prompt {index}", "natural_text": "a b c"}) + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    config = ParetoExperimentConfig(
        model_path="unused",
        secret_key="test-key",
        output_root=tmp_path / "outputs",
        experiment_id="processed-c4",
        calibration_samples=1,
        test_samples=1,
        exact_tokens=2,
        context_width=1,
        dataset_path=str(dataset_path),
    )

    result = build_manifest(config, Tokenizer(), model_max_length=16)
    rows = list(iter_jsonl(config.experiment_dir / "manifest.jsonl"))

    assert result == {"completed": 2, "filtered": 0}
    assert [row["natural_token_ids"] for row in rows] == [[0, 1], [0, 1]]
