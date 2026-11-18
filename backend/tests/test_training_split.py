from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml" / "scripts"))
from train_reranker import split_dataset


def test_training_and_validation_queries_do_not_overlap():
    samples = [SimpleNamespace(texts=[q, str(i)]) for q in ("Query A", "query a", "Query B", "Query C") for i in range(2)]
    train, validation = split_dataset(samples)
    assert train and validation
    assert {x.texts[0].casefold() for x in train}.isdisjoint({x.texts[0].casefold() for x in validation})
