import os
import pytest


@pytest.mark.skipif(not os.environ.get("LOCAL_OPT_MODEL"), reason="Set LOCAL_OPT_MODEL for the real local-model integration test")
def test_local_model_can_load() -> None:
    import torch
    from watermark.config import ModelConfig
    from utils.model import load_model_and_tokenizer
    model, tokenizer, device = load_model_and_tokenizer(ModelConfig(path=os.environ["LOCAL_OPT_MODEL"]))
    assert len(tokenizer) == model.config.vocab_size
    assert device.type == "cuda"
    assert next(model.parameters()).device.type == "cuda"
    del model
    torch.cuda.empty_cache()
