import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_episode_scorer.video import VideoSegment


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task: str
    timestamps: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    cameras: dict[str, VideoSegment]
    label: bool | None

    @property
    def runtime_seconds(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


@dataclass(frozen=True)
class LoadedDataset:
    repo_id: str
    root: Path
    camera_keys: list[str]
    nominal_runtime_seconds: float
    episodes: list[EpisodeRecord]


def load_episode_labels(repo_id: str, root: Path) -> dict[int, bool]:
    local_path = root / "results.json"
    if local_path.exists():
        results_path = local_path
    else:
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename="results.json"
            )
        except EntryNotFoundError:
            return {}
        results_path = Path(downloaded)

    raw_results = json.loads(results_path.read_text())
    return {int(row["episode"]): bool(row["success"]) for row in raw_results}


def normalize_camera_keys(
    requested_camera_keys: list[str], available_camera_keys: list[str]
) -> list[str]:
    if not requested_camera_keys:
        preferred_order = {
            "observation.images.top": 0,
            "observation.images.wrist": 1,
        }
        return sorted(available_camera_keys, key=lambda key: preferred_order.get(key, 99))

    resolved: list[str] = []
    for camera_key in requested_camera_keys:
        if camera_key in available_camera_keys:
            resolved.append(camera_key)
            continue

        full_key = f"observation.images.{camera_key}"
        if full_key not in available_camera_keys:
            raise ValueError(
                f"Camera key '{camera_key}' not found. Available: {', '.join(available_camera_keys)}"
            )
        resolved.append(full_key)

    return resolved


def load_lerobot_dataset(
    repo_id: str,
    root: Path | None,
    requested_camera_keys: list[str],
) -> LoadedDataset:
    dataset = LeRobotDataset(repo_id, root=root)
    label_by_episode = load_episode_labels(repo_id, dataset.root)
    if (
        label_by_episode
        and 0 not in label_by_episode
        and min(label_by_episode) == 1
        and max(label_by_episode) == dataset.meta.total_episodes
    ):
        label_by_episode = {
            episode_index - 1: label for episode_index, label in label_by_episode.items()
        }
    available_camera_keys = list(dataset.meta.video_keys)
    camera_keys = normalize_camera_keys(requested_camera_keys, available_camera_keys)

    task_by_index = {
        int(row.task_index): str(task_name) for task_name, row in dataset.meta.tasks.iterrows()
    }

    episodes: list[EpisodeRecord] = []
    for episode_index in range(dataset.meta.total_episodes):
        episode_meta = dataset.meta.episodes[episode_index]
        start_index = int(episode_meta["dataset_from_index"])
        end_index = int(episode_meta["dataset_to_index"])
        frame_slice = dataset.hf_dataset[start_index:end_index]

        task_index_value = frame_slice["task_index"][0]
        task_index = (
            int(task_index_value.item())
            if hasattr(task_index_value, "item")
            else int(task_index_value)
        )

        cameras: dict[str, VideoSegment] = {}
        for camera_key in camera_keys:
            cameras[camera_key] = VideoSegment(
                video_path=dataset.root
                / dataset.meta.video_path.format(
                    video_key=camera_key,
                    chunk_index=int(episode_meta[f"videos/{camera_key}/chunk_index"]),
                    file_index=int(episode_meta[f"videos/{camera_key}/file_index"]),
                ),
                from_timestamp=float(episode_meta[f"videos/{camera_key}/from_timestamp"]),
                to_timestamp=float(episode_meta[f"videos/{camera_key}/to_timestamp"]),
            )

        episodes.append(
            EpisodeRecord(
                episode_index=episode_index,
                task=task_by_index[task_index],
                timestamps=np.asarray(frame_slice["timestamp"], dtype=float),
                states=np.asarray(frame_slice["observation.state"], dtype=float),
                actions=np.asarray(frame_slice["action"], dtype=float),
                cameras=cameras,
                label=label_by_episode.get(episode_index),
            )
        )

    runtimes = [episode.runtime_seconds for episode in episodes]
    nominal_runtime_seconds = float(np.median(runtimes)) if runtimes else 0.0
    return LoadedDataset(
        repo_id=repo_id,
        root=Path(dataset.root),
        camera_keys=camera_keys,
        nominal_runtime_seconds=nominal_runtime_seconds,
        episodes=episodes,
    )
