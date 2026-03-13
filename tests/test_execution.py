import json
import tempfile
import unittest
from pathlib import Path

import torch

from lerobot_episode_scorer.cli import compute_summary
from lerobot_episode_scorer.execution import (
    DEFAULT_THRESHOLD,
    ExecutionCheckpoint,
    FeatureCacheIndex,
    FeatureIndexEntry,
    evaluate_execution_checkpoint,
    fit_execution_head,
    load_execution_checkpoint,
    load_dataset_manifest,
    load_feature_cache_index,
    save_execution_checkpoint,
    save_feature_cache_index,
)
from lerobot_episode_scorer.quality import score_visual_frame


class ExecutionPipelineTests(unittest.TestCase):
    def test_training_checkpoint_and_evaluation_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            feature_dir = Path(tempdir)
            cache_index = self._write_feature_cache(feature_dir)

            loaded_index = load_feature_cache_index(feature_dir)
            self.assertEqual(loaded_index.camera_keys, cache_index.camera_keys)
            self.assertEqual(len(loaded_index.records), len(cache_index.records))

            cache_index, head, calibrator, metrics = fit_execution_head(
                feature_dir=feature_dir,
                batch_size=2,
                num_epochs=6,
                learning_rate=1e-2,
                weight_decay=0.0,
                hidden_dim=16,
                dropout_rate=0.0,
                threshold=DEFAULT_THRESHOLD,
                device="cpu",
            )
            self.assertIn("calibrated_val_metrics", metrics)

            checkpoint_path = feature_dir / "checkpoint.pt"
            save_execution_checkpoint(
                path=checkpoint_path,
                head=head,
                cache_index=cache_index,
                calibrator=calibrator,
                threshold=DEFAULT_THRESHOLD,
            )
            checkpoint = load_execution_checkpoint(checkpoint_path)
            self.assertEqual(checkpoint.backend_name, "frozen_vlm_probe")
            self.assertEqual(checkpoint.camera_keys, ["observation.images.top"])

            rows, summary = evaluate_execution_checkpoint(
                feature_dir=feature_dir,
                checkpoint=checkpoint,
                batch_size=2,
                device="cpu",
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(summary["backend_name"], "frozen_vlm_probe")
            self.assertIn("execution_metrics", summary)
            self.assertIn("quality_only_baseline_metrics", summary)

    def test_score_summary_includes_metrics(self) -> None:
        rows = [
            {
                "repo_id": "demo/repo",
                "dataset_family": "task_fail",
                "episode_index": 0,
                "task": "pick",
                "label": 1,
                "quality_score": 0.8,
                "execution_score": 0.9,
                "execution_probability": 0.9,
                "combined_score": 0.72,
                "runtime_seconds": 10.0,
                "quality_components": {
                    "visual_clarity": 0.7,
                    "smoothness": 0.8,
                    "path_efficiency": 0.8,
                    "collision": 1.0,
                    "joint_stability": 0.8,
                    "actuator_saturation": 0.8,
                    "runtime": 1.0,
                    "visual_clarity_by_camera": {"observation.images.top": 0.7},
                },
                "execution_backend": "none",
                "decoded_vote": None,
            },
            {
                "repo_id": "demo/repo",
                "dataset_family": "task_fail",
                "episode_index": 1,
                "task": "pick",
                "label": 0,
                "quality_score": 0.3,
                "execution_score": 0.2,
                "execution_probability": 0.2,
                "combined_score": 0.06,
                "runtime_seconds": 11.0,
                "quality_components": {
                    "visual_clarity": 0.2,
                    "smoothness": 0.3,
                    "path_efficiency": 0.3,
                    "collision": 0.4,
                    "joint_stability": 0.3,
                    "actuator_saturation": 0.3,
                    "runtime": 0.9,
                    "visual_clarity_by_camera": {"observation.images.top": 0.2},
                },
                "execution_backend": "none",
                "decoded_vote": None,
            },
        ]

        summary = compute_summary(
            rows=rows,
            execution_backend="none",
            nominal_runtime_seconds=10.0,
            camera_keys=["observation.images.top"],
            checkpoint=None,
            fallback_vlm_model_id="demo/model",
        )

        self.assertEqual(summary["labels_available"], 2)
        self.assertIn("quality_metrics", summary)
        self.assertIn("execution_metrics", summary)
        self.assertIn("combined_metrics", summary)

    def test_score_summary_uses_checkpoint_threshold(self) -> None:
        rows = [
            {
                "repo_id": "demo/repo",
                "dataset_family": "task_fail",
                "episode_index": 0,
                "task": "pick",
                "label": 1,
                "quality_score": 0.8,
                "execution_score": 0.6,
                "execution_probability": 0.6,
                "combined_score": 0.48,
                "runtime_seconds": 10.0,
                "quality_components": {
                    "visual_clarity": 0.7,
                    "smoothness": 0.8,
                    "path_efficiency": 0.8,
                    "collision": 1.0,
                    "joint_stability": 0.8,
                    "actuator_saturation": 0.8,
                    "runtime": 1.0,
                    "visual_clarity_by_camera": {"observation.images.top": 0.7},
                },
                "execution_backend": "frozen_vlm_probe",
                "decoded_vote": None,
            },
            {
                "repo_id": "demo/repo",
                "dataset_family": "task_fail",
                "episode_index": 1,
                "task": "pick",
                "label": 0,
                "quality_score": 0.3,
                "execution_score": 0.4,
                "execution_probability": 0.4,
                "combined_score": 0.12,
                "runtime_seconds": 11.0,
                "quality_components": {
                    "visual_clarity": 0.2,
                    "smoothness": 0.3,
                    "path_efficiency": 0.3,
                    "collision": 0.4,
                    "joint_stability": 0.3,
                    "actuator_saturation": 0.3,
                    "runtime": 0.9,
                    "visual_clarity_by_camera": {"observation.images.top": 0.2},
                },
                "execution_backend": "frozen_vlm_probe",
                "decoded_vote": None,
            },
        ]
        checkpoint = ExecutionCheckpoint(
            checkpoint_path=Path("checkpoint.pt"),
            backend_name="frozen_vlm_probe",
            backend_version=1,
            vlm_model_id="demo/model",
            camera_keys=["observation.images.top"],
            frames_per_camera=4,
            hook_layer_indices=[1, 2, 3],
            feature_dim=4,
            hidden_dim=8,
            dropout_rate=0.0,
            threshold=0.8,
            calibration_scale=1.0,
            calibration_bias=0.0,
            label_mapping={"fail": 0, "success": 1},
            head_state_dict={},
        )

        summary = compute_summary(
            rows=rows,
            execution_backend="frozen_vlm_probe",
            nominal_runtime_seconds=10.0,
            camera_keys=["observation.images.top"],
            checkpoint=checkpoint,
            fallback_vlm_model_id="demo/model",
        )

        self.assertEqual(summary["execution_metrics"]["threshold"], 0.8)
        self.assertEqual(
            summary["dataset_family_summaries"]["task_fail"]["execution_metrics"]["threshold"],
            0.8,
        )

    def test_visual_score_responds_to_detail_and_exposure(self) -> None:
        flat_bright = torch.full((16, 16, 3), 255, dtype=torch.uint8).numpy()
        textured = torch.zeros((16, 16, 3), dtype=torch.uint8)
        textured[:, :8] = 40
        textured[:, 8:] = 200
        textured_frame = textured.numpy()

        flat_score = score_visual_frame(flat_bright)["score"]
        textured_score = score_visual_frame(textured_frame)["score"]
        self.assertLess(flat_score, textured_score)

    def test_manifest_loader_supports_episode_ranges_and_derived_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "repo_id": "demo/repo",
                                "dataset_family": "task_fail",
                                "split": "val",
                                "episode_from": 10,
                                "episode_to": 15,
                                "derived_label": 0,
                                "label_rule": "explicit task-failure dataset => task failure",
                                "use_for_training": True,
                                "use_for_evaluation": False,
                            }
                        ]
                    }
                )
            )

            entries = load_dataset_manifest(manifest_path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].episode_from, 10)
            self.assertEqual(entries[0].episode_to, 15)
            self.assertEqual(entries[0].derived_label, 0)
            self.assertEqual(
                entries[0].label_rule,
                "explicit task-failure dataset => task failure",
            )

    def _write_feature_cache(self, feature_dir: Path) -> FeatureCacheIndex:
        feature_paths = feature_dir / "features"
        feature_paths.mkdir(parents=True, exist_ok=True)

        rows = [
            ("train", True, False, 1, 0.9),
            ("train", True, False, 0, 0.2),
            ("train", True, False, 1, 0.8),
            ("train", True, False, 0, 0.3),
            ("val", True, False, 1, 0.85),
            ("val", True, False, 0, 0.25),
            ("test", False, True, 1, 0.8),
            ("test", False, True, 0, 0.2),
        ]

        records: list[FeatureIndexEntry] = []
        for index, (split, use_for_training, use_for_evaluation, label, quality_score) in enumerate(
            rows,
            start=1,
        ):
            layers = []
            for layer_index in range(3):
                sign = 1.0 if label == 1 else -1.0
                layers.append(torch.full((5, 4), sign + (0.05 * layer_index), dtype=torch.float32))

            relative_path = Path("features") / f"episode_{index:03d}.pt"
            torch.save(
                {
                    "layer_features": layers,
                    "decoded_vote": "success" if label == 1 else "fail",
                },
                feature_dir / relative_path,
            )
            records.append(
                FeatureIndexEntry(
                    feature_path=str(relative_path),
                    repo_id="demo/repo",
                    dataset_family="task_fail",
                    split=split,
                    use_for_training=use_for_training,
                    use_for_evaluation=use_for_evaluation,
                    episode_index=index,
                    task="pick and place",
                    label=label,
                    quality_score=quality_score,
                    decoded_vote="success" if label == 1 else "fail",
                    num_tokens=5,
                )
            )

        cache_index = FeatureCacheIndex(
            version=1,
            vlm_model_id="demo/model",
            camera_keys=["observation.images.top"],
            frames_per_camera=4,
            hook_layer_indices=[10, 20, 30],
            feature_dim=4,
            records=records,
        )
        save_feature_cache_index(feature_dir, cache_index)
        return cache_index


if __name__ == "__main__":
    unittest.main()
