"""
Train Local Cross-Encoder Reranker
====================================

Trains a CrossEncoder model on the labeled reranker dataset for
query-paper relevance scoring.

Usage:
    cd ml
    python scripts/train_reranker.py
    python scripts/train_reranker.py --epochs 5 --output models/reranker-v1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "reranker_dataset.jsonl"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "models" / "reranker-v1"
_DEFAULT_BASE_MODEL = "BAAI/bge-reranker-base"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load JSONL dataset."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows.append(row)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", line_num, exc)
    return rows


def prepare_training_samples(rows: list[dict[str, Any]]) -> list:
    """Convert dataset rows to InputExample format for CrossEncoder."""
    try:
        from sentence_transformers import InputExample
    except ImportError:
        logger.error("sentence-transformers is required. Install with: pip install sentence-transformers")
        sys.exit(1)

    samples = []
    for row in rows:
        query = row.get("query", "").strip()
        title = row.get("title", "").strip()
        abstract = row.get("abstract", "").strip()
        source = row.get("source", "").strip()
        label = float(row.get("label", 0))

        # Build candidate text: title + abstract + source
        parts = [title]
        if abstract:
            parts.append(abstract)
        if source:
            parts.append(f"Source: {source}")
        candidate_text = " ".join(parts)

        # Normalize label to [0, 1] range: 0→0.0, 1→0.5, 2→1.0
        normalized_label = label / 2.0

        samples.append(InputExample(texts=[query, candidate_text], label=normalized_label))

    return samples


def split_dataset(samples: list, train_ratio: float = 0.8) -> tuple[list, list]:
    """Split samples into train/validation sets."""
    import random
    # Split by research question, never by individual query-document pairs.
    groups = sorted({sample.texts[0].strip().casefold() for sample in samples})
    if len(groups) < 2:
        raise ValueError("At least two distinct queries are required for a held-out validation split")
    random.Random(42).shuffle(groups)
    split_idx = max(1, min(len(groups) - 1, int(len(groups) * train_ratio)))
    train_queries = set(groups[:split_idx])
    return ([s for s in samples if s.texts[0].strip().casefold() in train_queries],
            [s for s in samples if s.texts[0].strip().casefold() not in train_queries])



def train(
    dataset_path: Path,
    output_dir: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    warmup_ratio: float,
    learning_rate: float,
) -> dict[str, Any]:
    """Train the cross-encoder model."""
    try:
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error("Dependencies missing. Install: pip install sentence-transformers torch")
        sys.exit(1)

    # Load and prepare data
    rows = load_dataset(dataset_path)
    logger.info("Loaded %d rows from %s", len(rows), dataset_path)

    label_dist = {}
    for row in rows:
        l = int(row.get("label", 0))
        label_dist[l] = label_dist.get(l, 0) + 1
    logger.info("Label distribution: %s", label_dist)

    samples = prepare_training_samples(rows)
    train_samples, val_samples = split_dataset(samples)
    logger.info("Train: %d, Validation: %d", len(train_samples), len(val_samples))

    # Create train dataloader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

    # Create evaluator
    val_sentence_pairs = [[s.texts[0], s.texts[1]] for s in val_samples]
    val_scores = [s.label for s in val_samples]
    evaluator = CrossEncoderCorrelationEvaluator(val_sentence_pairs, val_scores, name="val")

    # Initialize model
    model = CrossEncoder(base_model, num_labels=1, max_length=512)
    logger.info("Base model: %s", base_model)

    # Warmup steps
    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)

    # Train
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        optimizer_params={"lr": learning_rate},
        show_progress_bar=True,
    )

    # fit() saves the best checkpoint selected by the held-out evaluator.
    logger.info("Best validation checkpoint saved to %s", output_dir)

    # Save metadata
    metadata = {
        "model_name": "local-reranker",
        "model_version": "v1",
        "model_type": "cross_encoder",
        "base_model": base_model,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
        "dataset_version": "v1",
        "dataset_path": str(dataset_path),
        "dataset_size": len(rows),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "label_distribution": label_dist,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "max_length": 512,
        },
        "label_schema": {
            "0": "irrelevant",
            "1": "partially_relevant",
            "2": "highly_relevant",
        },
        "metrics": None,
        "notes": "Metrics placeholder — run evaluate_reranker.py to compute actual metrics.",
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata saved to %s", metadata_path)

    logger.info("Model saved to %s", output_dir)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train local cross-encoder reranker")
    parser.add_argument("--dataset", type=str, default=str(_DEFAULT_DATASET), help="Path to JSONL dataset")
    parser.add_argument("--output", type=str, default=str(_DEFAULT_OUTPUT), help="Output model directory")
    parser.add_argument("--base-model", type=str, default=_DEFAULT_BASE_MODEL, help="HuggingFace base model")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    metadata = train(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.lr,
    )

    print(f"\n[OK] Training complete.")
    print(f"     Model: {args.output}")
    print(f"     Dataset: {metadata['dataset_size']} rows (train={metadata['train_size']}, val={metadata['val_size']})")
    print(f"     Labels: {metadata['label_distribution']}")
    print(f"     Run evaluate_reranker.py to compute metrics.")


if __name__ == "__main__":
    main()