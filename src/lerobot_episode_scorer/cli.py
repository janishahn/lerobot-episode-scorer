import argparse
import csv
from pathlib import Path

from lerobot_episode_scorer.dataset import load_lerobot_dataset
from lerobot_episode_scorer.execution import (
    DEFAULT_THRESHOLD,
    DEFAULT_VLM_MODEL_ID,
    ExecutionCheckpoint,
    FrozenVLMProbeExecutionScorer,
    default_camera_keys_for_backend,
    load_execution_checkpoint,
    write_json,
)
from lerobot_episode_scorer.metrics import compute_binary_metrics, sanitized_metrics
from lerobot_episode_scorer.quality import EpisodeQualityScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score LeRobot episodes with quality metrics and an optional execution backend."
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id.")
    parser.add_argument("--root", type=Path, default=None, help="Optional local LeRobot root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for score outputs.",
    )
    parser.add_argument(
        "--dataset-family",
        default="custom",
        help="Dataset family name stored in the outputs.",
    )
    parser.add_argument(
        "--camera-key",
        action="append",
        default=[],
        help="Camera key to score. Use bare names like 'top' and 'wrist' or full feature keys.",
    )
    parser.add_argument(
        "--nominal-runtime-seconds",
        type=float,
        default=None,
        help="Runtime target used for runtime scoring. Defaults to the dataset median runtime.",
    )
    parser.add_argument(
        "--execution-backend",
        choices=["none", "frozen_vlm_probe"],
        default="none",
        help="Execution backend used to score task success.",
    )
    parser.add_argument(
        "--execution-checkpoint",
        type=Path,
        default=None,
        help="Native checkpoint path for --execution-backend frozen_vlm_probe.",
    )
    parser.add_argument(
        "--vlm-model-id",
        default=DEFAULT_VLM_MODEL_ID,
        help="Only written to summary metadata for backend='none'; backend checkpoints carry their own model id.",
    )
    parser.add_argument("--device", default=None, help="Torch device for execution scoring.")
    return parser


def flatten_episode_row(row: dict[str, object]) -> dict[str, object]:
    quality_components = row["quality_components"]
    flat_row = {
        "repo_id": row["repo_id"],
        "dataset_family": row["dataset_family"],
        "episode_index": row["episode_index"],
        "task": row["task"],
        "label": row["label"],
        "quality_score": row["quality_score"],
        "execution_score": row["execution_score"],
        "execution_probability": row["execution_probability"],
        "combined_score": row["combined_score"],
        "runtime_seconds": row["runtime_seconds"],
        "execution_backend": row["execution_backend"],
        "decoded_vote": row["decoded_vote"],
        "visual_clarity": quality_components["visual_clarity"],
        "smoothness": quality_components["smoothness"],
        "path_efficiency": quality_components["path_efficiency"],
        "collision": quality_components["collision"],
        "joint_stability": quality_components["joint_stability"],
        "actuator_saturation": quality_components["actuator_saturation"],
        "runtime": quality_components["runtime"],
    }

    for camera_key, score in quality_components["visual_clarity_by_camera"].items():
        flat_row[f"{camera_key.replace('.', '_')}_visual"] = score

    return flat_row


def compute_family_summaries(
    rows: list[dict[str, object]],
    execution_threshold: float,
) -> dict[str, dict[str, object]]:
    families = sorted({str(row["dataset_family"]) for row in rows})
    family_summaries: dict[str, dict[str, object]] = {}
    for family in families:
        family_rows = [row for row in rows if row["dataset_family"] == family]
        labels = [int(row["label"]) for row in family_rows if row["label"] is not None]
        family_summary: dict[str, object] = {
            "count": len(family_rows),
            "quality_mean": sum(float(row["quality_score"]) for row in family_rows)
            / len(family_rows)
            if family_rows
            else 0.0,
            "execution_mean": sum(float(row["execution_score"]) for row in family_rows)
            / len(family_rows)
            if family_rows
            else 0.0,
            "combined_mean": sum(float(row["combined_score"]) for row in family_rows)
            / len(family_rows)
            if family_rows
            else 0.0,
        }
        if labels:
            labeled_rows = [row for row in family_rows if row["label"] is not None]
            family_summary["quality_metrics"] = sanitized_metrics(
                compute_binary_metrics(
                    [float(row["quality_score"]) for row in labeled_rows],
                    [int(row["label"]) for row in labeled_rows],
                )
            )
            family_summary["execution_metrics"] = sanitized_metrics(
                compute_binary_metrics(
                    [float(row["execution_probability"]) for row in labeled_rows],
                    [int(row["label"]) for row in labeled_rows],
                    threshold=execution_threshold,
                )
            )
            family_summary["combined_metrics"] = sanitized_metrics(
                compute_binary_metrics(
                    [float(row["combined_score"]) for row in labeled_rows],
                    [int(row["label"]) for row in labeled_rows],
                    threshold=execution_threshold,
                )
            )
        family_summaries[family] = family_summary
    return family_summaries


def compute_summary(
    rows: list[dict[str, object]],
    execution_backend: str,
    nominal_runtime_seconds: float,
    camera_keys: list[str],
    checkpoint: ExecutionCheckpoint | None,
    fallback_vlm_model_id: str,
) -> dict[str, object]:
    execution_threshold = DEFAULT_THRESHOLD if checkpoint is None else checkpoint.threshold
    labels = [int(row["label"]) for row in rows if row["label"] is not None]
    if rows:
        quality_mean = sum(float(row["quality_score"]) for row in rows) / len(rows)
        execution_mean = sum(float(row["execution_score"]) for row in rows) / len(rows)
        combined_mean = sum(float(row["combined_score"]) for row in rows) / len(rows)
    else:
        quality_mean = 0.0
        execution_mean = 0.0
        combined_mean = 0.0
    summary: dict[str, object] = {
        "total_episodes": len(rows),
        "labels_available": len(labels),
        "execution_backend": execution_backend,
        "quality_mean": quality_mean,
        "execution_mean": execution_mean,
        "combined_mean": combined_mean,
        "camera_keys": camera_keys,
        "nominal_runtime_seconds": nominal_runtime_seconds,
        "dataset_family_summaries": compute_family_summaries(rows, execution_threshold),
        "vlm_model_id": checkpoint.vlm_model_id
        if checkpoint is not None
        else fallback_vlm_model_id,
        "execution_checkpoint": None if checkpoint is None else str(checkpoint.checkpoint_path),
    }

    if labels:
        labeled_rows = [row for row in rows if row["label"] is not None]
        quality_probabilities = [float(row["quality_score"]) for row in labeled_rows]
        execution_probabilities = [float(row["execution_probability"]) for row in labeled_rows]
        combined_probabilities = [float(row["combined_score"]) for row in labeled_rows]
        label_values = [int(row["label"]) for row in labeled_rows]
        summary["quality_metrics"] = sanitized_metrics(
            compute_binary_metrics(quality_probabilities, label_values)
        )
        summary["execution_metrics"] = sanitized_metrics(
            compute_binary_metrics(
                execution_probabilities,
                label_values,
                threshold=execution_threshold,
            )
        )
        summary["combined_metrics"] = sanitized_metrics(
            compute_binary_metrics(
                combined_probabilities,
                label_values,
                threshold=execution_threshold,
            )
        )

    return summary


def write_outputs(
    output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "episode_scores.json", rows)
    write_json(output_dir / "summary.json", summary)

    flat_rows = [flatten_episode_row(row) for row in rows]
    if not flat_rows:
        return
    with (output_dir / "episode_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = None
    if args.execution_backend == "frozen_vlm_probe":
        if args.execution_checkpoint is None:
            raise ValueError(
                "--execution-checkpoint is required for --execution-backend frozen_vlm_probe"
            )
        checkpoint = load_execution_checkpoint(args.execution_checkpoint)

    requested_camera_keys = default_camera_keys_for_backend(args.camera_key, checkpoint)
    loaded_dataset = load_lerobot_dataset(args.repo_id, args.root, requested_camera_keys)
    nominal_runtime_seconds = (
        args.nominal_runtime_seconds
        if args.nominal_runtime_seconds is not None
        else loaded_dataset.nominal_runtime_seconds
    )
    print(
        f"Loaded {len(loaded_dataset.episodes)} episodes from {loaded_dataset.repo_id} "
        f"with cameras {loaded_dataset.camera_keys}"
    )

    quality_scorer = EpisodeQualityScorer(nominal_runtime_seconds=nominal_runtime_seconds)
    execution_scorer = None
    if checkpoint is not None:
        execution_scorer = FrozenVLMProbeExecutionScorer(checkpoint=checkpoint, device=args.device)

    rows: list[dict[str, object]] = []
    try:
        for index, episode in enumerate(loaded_dataset.episodes, start=1):
            print(f"Scoring episode {index}/{len(loaded_dataset.episodes)}", flush=True)
            quality = quality_scorer.score_episode(episode)
            if execution_scorer is None:
                execution_score = 1.0
                execution_probability = 1.0
                decoded_vote = None
            else:
                execution = execution_scorer.score_episode(episode)
                execution_score = execution.score
                execution_probability = execution.probability
                decoded_vote = execution.decoded_vote

            rows.append(
                {
                    "repo_id": loaded_dataset.repo_id,
                    "dataset_family": args.dataset_family,
                    "episode_index": episode.episode_index,
                    "task": episode.task,
                    "label": episode.label,
                    "quality_score": quality["aggregate"],
                    "execution_score": execution_score,
                    "execution_probability": execution_probability,
                    "combined_score": quality["aggregate"] * execution_score,
                    "runtime_seconds": quality["runtime_seconds"],
                    "quality_components": quality,
                    "execution_backend": args.execution_backend,
                    "decoded_vote": decoded_vote,
                }
            )
    finally:
        if execution_scorer is not None:
            execution_scorer.cleanup()

    summary = compute_summary(
        rows=rows,
        execution_backend=args.execution_backend,
        nominal_runtime_seconds=nominal_runtime_seconds,
        camera_keys=loaded_dataset.camera_keys,
        checkpoint=checkpoint,
        fallback_vlm_model_id=args.vlm_model_id,
    )
    summary["repo_id"] = loaded_dataset.repo_id
    summary["dataset_family"] = args.dataset_family
    write_outputs(args.output_dir, rows, summary)


if __name__ == "__main__":
    main()
