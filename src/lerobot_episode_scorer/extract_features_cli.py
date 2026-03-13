import argparse
from datetime import UTC, datetime
from pathlib import Path

import torch

from lerobot_episode_scorer.dataset import load_lerobot_dataset
from lerobot_episode_scorer.execution import (
    DEFAULT_FRAMES_PER_CAMERA,
    DEFAULT_VLM_MODEL_ID,
    FeatureCacheIndex,
    FeatureIndexEntry,
    VLMFeatureExtractor,
    load_dataset_manifest,
    normalize_label,
    save_feature_cache_index,
    write_json,
)
from lerobot_episode_scorer.quality import EpisodeQualityScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract frozen VLM features for labeled or unlabeled LeRobot episodes."
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
        help="JSON file describing the datasets to extract.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the feature cache will be written.",
    )
    parser.add_argument(
        "--camera-key",
        action="append",
        default=[],
        help="Camera key to score. Use bare names like 'top' and 'wrist' or full feature keys.",
    )
    parser.add_argument(
        "--frames-per-camera",
        type=int,
        default=DEFAULT_FRAMES_PER_CAMERA,
        help="Number of uniformly sampled frames per camera.",
    )
    parser.add_argument(
        "--vlm-model-id",
        default=DEFAULT_VLM_MODEL_ID,
        help="Adapter-backed VLM used to extract frozen execution features.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Write progress and print a status line every N extracted episodes.",
    )
    parser.add_argument("--device", default=None, help="Torch device for feature extraction.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_entries = load_dataset_manifest(args.dataset_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "features").mkdir(parents=True, exist_ok=True)
    run_started_at = datetime.now(UTC).isoformat()
    write_json(
        args.output_dir / "run.json",
        {
            "command": "combined-episode-extract-features",
            "started_at": run_started_at,
            "dataset_manifest": str(args.dataset_manifest),
            "output_dir": str(args.output_dir),
            "camera_keys": args.camera_key,
            "frames_per_camera": args.frames_per_camera,
            "vlm_model_id": args.vlm_model_id,
            "device": args.device,
            "log_every": args.log_every,
        },
    )

    extractor = VLMFeatureExtractor(vlm_model_id=args.vlm_model_id, device=args.device)
    index_entries: list[FeatureIndexEntry] = []
    feature_dim: int | None = None
    camera_keys: list[str] | None = None
    record_index = 0
    total_manifest_episodes = 0
    completed_manifest_episodes = 0
    per_dataset_plan: list[dict[str, object]] = []

    for manifest_entry in manifest_entries:
        if manifest_entry.episode_to is None:
            planned_count = None
        else:
            planned_count = manifest_entry.episode_to - manifest_entry.episode_from
            total_manifest_episodes += planned_count
        per_dataset_plan.append(
            {
                "repo_id": manifest_entry.repo_id,
                "dataset_family": manifest_entry.dataset_family,
                "split": manifest_entry.split,
                "episode_from": manifest_entry.episode_from,
                "episode_to": manifest_entry.episode_to,
                "planned_count": planned_count,
                "derived_label": manifest_entry.derived_label,
                "label_rule": manifest_entry.label_rule,
            }
        )

    write_json(
        args.output_dir / "progress.json",
        {
            "command": "combined-episode-extract-features",
            "started_at": run_started_at,
            "status": "running",
            "completed_episodes": 0,
            "planned_episodes": total_manifest_episodes,
            "current_dataset": None,
            "dataset_plan": per_dataset_plan,
        },
    )

    try:
        for manifest_entry in manifest_entries:
            loaded_dataset = load_lerobot_dataset(
                repo_id=manifest_entry.repo_id,
                root=manifest_entry.root,
                requested_camera_keys=args.camera_key,
            )
            quality_scorer = EpisodeQualityScorer(
                nominal_runtime_seconds=loaded_dataset.nominal_runtime_seconds
            )
            print(
                f"Extracting {len(loaded_dataset.episodes)} episodes from {loaded_dataset.repo_id} "
                f"with cameras {loaded_dataset.camera_keys}"
            )
            if camera_keys is None:
                camera_keys = loaded_dataset.camera_keys
            elif loaded_dataset.camera_keys != camera_keys:
                raise ValueError(
                    "All datasets in one feature cache must use the same camera keys. "
                    f"Expected {camera_keys}, got {loaded_dataset.camera_keys} for "
                    f"{loaded_dataset.repo_id}."
                )

            episode_stop = (
                len(loaded_dataset.episodes)
                if manifest_entry.episode_to is None
                else manifest_entry.episode_to
            )
            selected_episodes = loaded_dataset.episodes[manifest_entry.episode_from : episode_stop]
            selected_count = len(selected_episodes)
            if not selected_episodes:
                raise ValueError(
                    f"No episodes selected for {loaded_dataset.repo_id} split={manifest_entry.split} "
                    f"range=[{manifest_entry.episode_from}, {episode_stop})."
                )
            print(
                f"Selected {selected_count} episodes for split={manifest_entry.split} "
                f"range=[{manifest_entry.episode_from}, {episode_stop}) "
                f"label={manifest_entry.derived_label}"
            )
            write_json(
                args.output_dir / "progress.json",
                {
                    "command": "combined-episode-extract-features",
                    "started_at": run_started_at,
                    "status": "running",
                    "completed_episodes": completed_manifest_episodes,
                    "planned_episodes": total_manifest_episodes,
                    "current_dataset": {
                        "repo_id": loaded_dataset.repo_id,
                        "dataset_family": manifest_entry.dataset_family,
                        "split": manifest_entry.split,
                        "episode_from": manifest_entry.episode_from,
                        "episode_to": episode_stop,
                        "selected_count": selected_count,
                        "derived_label": manifest_entry.derived_label,
                        "label_rule": manifest_entry.label_rule,
                    },
                    "dataset_plan": per_dataset_plan,
                },
            )

            for episode_offset, episode in enumerate(selected_episodes, start=1):
                record_index += 1
                quality = quality_scorer.score_episode(episode)
                layer_features, decoded_vote = extractor.extract_episode_features(
                    episode=episode,
                    frames_per_camera=args.frames_per_camera,
                )
                if feature_dim is None:
                    feature_dim = int(layer_features[0].shape[1])

                feature_path = Path("features") / f"episode_{record_index:06d}.pt"
                torch.save(
                    {
                        "layer_features": layer_features,
                        "decoded_vote": decoded_vote,
                    },
                    args.output_dir / feature_path,
                )

                index_entries.append(
                    FeatureIndexEntry(
                        feature_path=str(feature_path),
                        repo_id=loaded_dataset.repo_id,
                        dataset_family=manifest_entry.dataset_family,
                        split=manifest_entry.split,
                        use_for_training=manifest_entry.use_for_training,
                        use_for_evaluation=manifest_entry.use_for_evaluation,
                        episode_index=episode.episode_index,
                        task=episode.task,
                        label=(
                            manifest_entry.derived_label
                            if manifest_entry.derived_label is not None
                            else normalize_label(episode.label)
                        ),
                        quality_score=float(quality["aggregate"]),
                        decoded_vote=decoded_vote,
                        num_tokens=int(layer_features[0].shape[0]),
                    )
                )
                completed_manifest_episodes += 1
                if (
                    episode_offset == selected_count
                    or completed_manifest_episodes == 1
                    or completed_manifest_episodes % args.log_every == 0
                ):
                    print(
                        f"Extracted {completed_manifest_episodes}"
                        + (f"/{total_manifest_episodes}" if total_manifest_episodes > 0 else "")
                        + f" episodes; current dataset progress {episode_offset}/{selected_count}"
                    )
                    write_json(
                        args.output_dir / "progress.json",
                        {
                            "command": "combined-episode-extract-features",
                            "started_at": run_started_at,
                            "status": "running",
                            "completed_episodes": completed_manifest_episodes,
                            "planned_episodes": total_manifest_episodes,
                            "current_dataset": {
                                "repo_id": loaded_dataset.repo_id,
                                "dataset_family": manifest_entry.dataset_family,
                                "split": manifest_entry.split,
                                "episode_from": manifest_entry.episode_from,
                                "episode_to": episode_stop,
                                "selected_count": selected_count,
                                "completed_count": episode_offset,
                                "derived_label": manifest_entry.derived_label,
                                "label_rule": manifest_entry.label_rule,
                            },
                            "dataset_plan": per_dataset_plan,
                        },
                    )
    except Exception as error:
        write_json(
            args.output_dir / "progress.json",
            {
                "command": "combined-episode-extract-features",
                "started_at": run_started_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "status": "failed",
                "completed_episodes": completed_manifest_episodes,
                "planned_episodes": total_manifest_episodes,
                "current_dataset": None,
                "dataset_plan": per_dataset_plan,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )
        raise
    finally:
        extractor.cleanup()

    if feature_dim is None:
        raise ValueError("Feature extraction produced no records")

    cache_index = FeatureCacheIndex(
        version=1,
        vlm_model_id=args.vlm_model_id,
        camera_keys=camera_keys or [],
        frames_per_camera=args.frames_per_camera,
        hook_layer_indices=extractor.hook_layer_indices,
        feature_dim=feature_dim,
        records=index_entries,
    )
    save_feature_cache_index(args.output_dir, cache_index)
    write_json(
        args.output_dir / "manifest.json",
        {
            "datasets": [
                {
                    "repo_id": entry.repo_id,
                    "dataset_family": entry.dataset_family,
                    "split": entry.split,
                    "root": None if entry.root is None else str(entry.root),
                    "episode_from": entry.episode_from,
                    "episode_to": entry.episode_to,
                    "derived_label": entry.derived_label,
                    "label_rule": entry.label_rule,
                    "use_for_training": entry.use_for_training,
                    "use_for_evaluation": entry.use_for_evaluation,
                }
                for entry in manifest_entries
            ]
        },
    )
    labeled_records = [entry for entry in index_entries if entry.label is not None]
    success_count = sum(entry.label == 1 for entry in labeled_records)
    failure_count = sum(entry.label == 0 for entry in labeled_records)
    split_counts = {
        split: sum(entry.split == split for entry in index_entries)
        for split in sorted({entry.split for entry in index_entries})
    }
    family_counts = {
        family: sum(entry.dataset_family == family for entry in index_entries)
        for family in sorted({entry.dataset_family for entry in index_entries})
    }
    summary = {
        "command": "combined-episode-extract-features",
        "started_at": run_started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "output_dir": str(args.output_dir),
        "record_count": len(index_entries),
        "labeled_record_count": len(labeled_records),
        "success_count": success_count,
        "failure_count": failure_count,
        "split_counts": split_counts,
        "dataset_family_counts": family_counts,
        "camera_keys": cache_index.camera_keys,
        "frames_per_camera": cache_index.frames_per_camera,
        "vlm_model_id": cache_index.vlm_model_id,
        "feature_dim": cache_index.feature_dim,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "progress.json",
        {
            "command": "combined-episode-extract-features",
            "started_at": run_started_at,
            "completed_at": summary["completed_at"],
            "status": "completed",
            "completed_episodes": len(index_entries),
            "planned_episodes": total_manifest_episodes,
            "current_dataset": None,
            "dataset_plan": per_dataset_plan,
            "summary": summary,
        },
    )
    print(
        f"Finished extracting {len(index_entries)} episodes to {args.output_dir} "
        f"(success={success_count}, failure={failure_count})"
    )


if __name__ == "__main__":
    main()
