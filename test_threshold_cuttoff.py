from kenlm_manager import *
import pytest


load_kenlm_model(src_lang='vi', pretrained_models=['wikipedia'])
wikipedia_kenml = KenlmModel(model_dataset="/root/.cache/wikipedia", language="yo")


@pytest.mark.parametrize("key, expected",
                         [("google", 6), ("youtube", 7)])
def test_cache_value(web_cache, key, expected):
    assert web_cache.get(key) == expected