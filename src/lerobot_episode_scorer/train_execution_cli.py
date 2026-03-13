import argparse
from datetime import UTC, datetime
from pathlib import Path

from lerobot_episode_scorer.execution import (
    DEFAULT_THRESHOLD,
    filter_entries_for_training,
    fit_execution_head,
    load_feature_cache_index,
    save_execution_checkpoint,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a frozen-VLM execution head from a feature cache."
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Directory produced by combined-episode-extract-features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where checkpoints and metrics will be written.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Per-layer head hidden size.")
    parser.add_argument(
        "--dropout-rate", type=float, default=0.3, help="Dropout used in the heads."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Decision threshold stored with the checkpoint.",
    )
    parser.add_argument("--device", default=None, help="Torch device for training.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = datetime.now(UTC).isoformat()
    cache_index = load_feature_cache_index(args.feature_dir)
    train_entries, val_entries = filter_entries_for_training(cache_index)
    train_positive_count = sum(entry.label == 1 for entry in train_entries)
    train_negative_count = sum(entry.label == 0 for entry in train_entries)
    write_json(
        args.output_dir / "run.json",
        {
            "command": "combined-episode-train-execution",
            "started_at": run_started_at,
            "feature_dir": str(args.feature_dir),
            "output_dir": str(args.output_dir),
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout_rate": args.dropout_rate,
            "threshold": args.threshold,
            "device": args.device,
            "train_count": len(train_entries),
            "val_count": len(val_entries),
            "train_positive_count": train_positive_count,
            "train_negative_count": train_negative_count,
            "camera_keys": cache_index.camera_keys,
            "vlm_model_id": cache_index.vlm_model_id,
        },
    )
    write_json(
        args.output_dir / "progress.json",
        {
            "command": "combined-episode-train-execution",
            "started_at": run_started_at,
            "status": "running",
            "completed_epochs": 0,
            "num_epochs": args.num_epochs,
            "train_count": len(train_entries),
            "val_count": len(val_entries),
            "train_positive_count": train_positive_count,
            "train_negative_count": train_negative_count,
            "latest_metrics": None,
            "history": [],
        },
    )
    print(
        f"Training execution head from {args.feature_dir}: "
        f"train={len(train_entries)} (pos={train_positive_count}, neg={train_negative_count}), "
        f"val={len(val_entries)}, epochs={args.num_epochs}"
    )

    def on_epoch_end(progress: dict[str, object]) -> None:
        metrics = progress["metrics"]
        print(
            f"Epoch {progress['epoch']}/{args.num_epochs}: "
            f"train_loss={metrics['train_loss']:.4f} "
            f"auroc={metrics['auroc']} "
            f"balanced_accuracy={metrics['balanced_accuracy']} "
            f"f1={metrics['f1']}"
        )
        write_json(
            args.output_dir / "progress.json",
            {
                "command": "combined-episode-train-execution",
                "started_at": run_started_at,
                "status": "running",
                "completed_epochs": progress["epoch"],
                "num_epochs": args.num_epochs,
                "train_count": progress["train_count"],
                "val_count": progress["val_count"],
                "train_positive_count": progress["positive_count"],
                "train_negative_count": progress["negative_count"],
                "latest_metrics": metrics,
                "history": progress["history"],
            },
        )

    try:
        cache_index, head, calibrator, metrics = fit_execution_head(
            feature_dir=args.feature_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            threshold=args.threshold,
            device=args.device,
            on_epoch_end=on_epoch_end,
        )
    except Exception as error:
        write_json(
            args.output_dir / "progress.json",
            {
                "command": "combined-episode-train-execution",
                "started_at": run_started_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "status": "failed",
                "completed_epochs": 0,
                "num_epochs": args.num_epochs,
                "train_count": len(train_entries),
                "val_count": len(val_entries),
                "train_positive_count": train_positive_count,
                "train_negative_count": train_negative_count,
                "latest_metrics": None,
                "history": [],
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )
        raise

    save_execution_checkpoint(
        path=args.output_dir / "checkpoint.pt",
        head=head,
        cache_index=cache_index,
        calibrator=calibrator,
        threshold=args.threshold,
    )
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(
        args.output_dir / "progress.json",
        {
            "command": "combined-episode-train-execution",
            "started_at": run_started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "status": "completed",
            "completed_epochs": args.num_epochs,
            "num_epochs": args.num_epochs,
            "train_count": len(train_entries),
            "val_count": len(val_entries),
            "train_positive_count": train_positive_count,
            "train_negative_count": train_negative_count,
            "latest_metrics": metrics["calibrated_val_metrics"],
            "history": metrics["history"],
        },
    )
    print(
        f"Finished training execution head; wrote checkpoint to {args.output_dir / 'checkpoint.pt'}"
    )


if __name__ == "__main__":
    main()
