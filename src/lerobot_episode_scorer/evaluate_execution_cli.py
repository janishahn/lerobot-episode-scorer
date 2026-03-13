import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path

from lerobot_episode_scorer.execution import (
    evaluate_execution_checkpoint,
    load_execution_checkpoint,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained execution checkpoint on cached episode features."
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Directory produced by combined-episode-extract-features.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Execution checkpoint produced by combined-episode-train-execution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-episode outputs and summary metrics will be written.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--device", default=None, help="Torch device for evaluation.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = datetime.now(UTC).isoformat()
    write_json(
        args.output_dir / "run.json",
        {
            "command": "combined-episode-evaluate-execution",
            "started_at": run_started_at,
            "feature_dir": str(args.feature_dir),
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.output_dir),
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )

    checkpoint = load_execution_checkpoint(args.checkpoint)
    print(
        f"Evaluating {checkpoint.backend_name} on cached features from {args.feature_dir} "
        f"with checkpoint {args.checkpoint}"
    )
    try:
        rows, summary = evaluate_execution_checkpoint(
            feature_dir=args.feature_dir,
            checkpoint=checkpoint,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as error:
        write_json(
            args.output_dir / "progress.json",
            {
                "command": "combined-episode-evaluate-execution",
                "started_at": run_started_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "status": "failed",
                "count": 0,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )
        raise

    write_json(args.output_dir / "episode_scores.json", rows)
    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "episode_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output_dir / "progress.json",
        {
            "command": "combined-episode-evaluate-execution",
            "started_at": run_started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "status": "completed",
            "count": len(rows),
            "summary": summary,
        },
    )
    print(f"Finished evaluation for {len(rows)} episodes; wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
