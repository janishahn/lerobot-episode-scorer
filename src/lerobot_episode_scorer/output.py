import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from lerobot_episode_scorer.metrics import compute_binary_metrics, sanitized_metrics


def flatten_episode_row(row: dict) -> dict:
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
        "vlm_response": row.get("vlm_response"),
        "reasoning_trace": row.get("reasoning_trace"),
        "camera_used": row.get("camera_used"),
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


def compute_family_summaries(rows: list[dict]) -> dict[str, dict]:
    families = sorted({str(row["dataset_family"]) for row in rows})
    family_summaries: dict[str, dict] = {}
    for family in families:
        family_rows = [row for row in rows if row["dataset_family"] == family]
        labels = [int(row["label"]) for row in family_rows if row["label"] is not None]
        family_summary: dict = {
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
                )
            )
            family_summary["combined_metrics"] = sanitized_metrics(
                compute_binary_metrics(
                    [float(row["combined_score"]) for row in labeled_rows],
                    [int(row["label"]) for row in labeled_rows],
                )
            )
        family_summaries[family] = family_summary
    return family_summaries


def compute_summary(
    rows: list[dict],
    execution_backend: str,
    nominal_runtime_seconds: float,
    camera_keys: list[str],
    execution_model: str,
    repo_id: str,
    dataset_family: str,
) -> dict:
    labels = [int(row["label"]) for row in rows if row["label"] is not None]
    if rows:
        quality_mean = sum(float(row["quality_score"]) for row in rows) / len(rows)
        execution_mean = sum(float(row["execution_score"]) for row in rows) / len(rows)
        combined_mean = sum(float(row["combined_score"]) for row in rows) / len(rows)
    else:
        quality_mean = 0.0
        execution_mean = 0.0
        combined_mean = 0.0

    summary: dict = {
        "total_episodes": len(rows),
        "labels_available": len(labels),
        "execution_backend": execution_backend,
        "quality_mean": quality_mean,
        "execution_mean": execution_mean,
        "combined_mean": combined_mean,
        "camera_keys": camera_keys,
        "nominal_runtime_seconds": nominal_runtime_seconds,
        "dataset_family_summaries": compute_family_summaries(rows),
        "execution_model": execution_model,
        "repo_id": repo_id,
        "dataset_family": dataset_family,
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
            compute_binary_metrics(execution_probabilities, label_values)
        )
        summary["combined_metrics"] = sanitized_metrics(
            compute_binary_metrics(combined_probabilities, label_values)
        )

    return summary


@dataclass
class RollingOutputWriter:
    output_dir: Path
    execution_backend: str
    nominal_runtime_seconds: float
    camera_keys: list[str]
    execution_model: str
    repo_id: str
    dataset_family: str
    rows: list[dict] = field(default_factory=list)
    csv_writer: csv.DictWriter | None = field(default=None, repr=False)
    csv_handle: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_episode(self, row: dict) -> None:
        self.rows.append(row)
        self._write_episode_json()
        self._append_csv_row(row)
        self._write_summary_json()

    def _write_episode_json(self) -> None:
        path = self.output_dir / "episode_scores.json"
        path.write_text(json.dumps(self.rows, indent=2))

    def _write_summary_json(self) -> None:
        summary = compute_summary(
            rows=self.rows,
            execution_backend=self.execution_backend,
            nominal_runtime_seconds=self.nominal_runtime_seconds,
            camera_keys=self.camera_keys,
            execution_model=self.execution_model,
            repo_id=self.repo_id,
            dataset_family=self.dataset_family,
        )
        path = self.output_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2))

    def _append_csv_row(self, row: dict) -> None:
        path = self.output_dir / "episode_scores.csv"
        flat_row = flatten_episode_row(row)

        if self.csv_writer is None:
            self.csv_handle = path.open("w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=list(flat_row.keys()))
            self.csv_writer.writeheader()

        self.csv_writer.writerow(flat_row)

    def finalize(self) -> None:
        if self.csv_handle is not None:
            self.csv_handle.close()
