from dataclasses import dataclass

import numpy as np

from lerobot_episode_scorer.dataset import EpisodeRecord
from lerobot_episode_scorer.video import sample_segment_frames


@dataclass(frozen=True)
class EpisodeQualityScorer:
    nominal_runtime_seconds: float
    visual_samples_per_camera: int = 8

    def score_episode(self, episode: EpisodeRecord) -> dict[str, object]:
        visual_by_camera: dict[str, float] = {}
        visual_raw_by_camera: dict[str, dict[str, float]] = {}
        for camera_key, segment in episode.cameras.items():
            frames = sample_segment_frames(segment, self.visual_samples_per_camera)
            frame_scores = [score_visual_frame(frame) for frame in frames]
            visual_by_camera[camera_key] = float(
                np.mean([frame_score["score"] for frame_score in frame_scores])
            )
            visual_raw_by_camera[camera_key] = {
                "laplacian_log_variance": float(
                    np.mean([frame_score["laplacian_log_variance"] for frame_score in frame_scores])
                ),
                "contrast": float(
                    np.mean([frame_score["contrast"] for frame_score in frame_scores])
                ),
                "exposure": float(
                    np.mean([frame_score["exposure"] for frame_score in frame_scores])
                ),
            }

        scalar_scores = {
            "visual_clarity": float(np.mean(list(visual_by_camera.values()))),
            "smoothness": score_smoothness(episode.states, episode.timestamps),
            "path_efficiency": score_path_efficiency(episode.states),
            "collision": score_collision(episode.states, episode.timestamps),
            "joint_stability": score_joint_stability(episode.states, episode.timestamps),
            "actuator_saturation": score_actuator_saturation(episode.states, episode.actions),
            "runtime": score_runtime(episode.runtime_seconds, self.nominal_runtime_seconds),
        }

        weights = {
            "visual_clarity": 20.0,
            "smoothness": 10.0,
            "path_efficiency": 10.0,
            "collision": 10.0,
            "joint_stability": 10.0,
            "actuator_saturation": 10.0,
            "runtime": 20.0,
        }
        weighted_total = sum(scalar_scores[name] * weight for name, weight in weights.items())

        return {
            "aggregate": float(weighted_total / sum(weights.values())),
            "runtime_seconds": episode.runtime_seconds,
            "visual_clarity_by_camera": visual_by_camera,
            "visual_raw_by_camera": visual_raw_by_camera,
            **scalar_scores,
        }


def score_visual_frame(frame: np.ndarray) -> dict[str, float]:
    gray = frame.mean(axis=2).astype(np.float32) / 255.0
    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    laplacian_log_variance = float(np.log1p(laplacian.var()))
    contrast = float(gray.std())
    exposure = max(0.0, 1.0 - abs(float(gray.mean()) - 0.5) / 0.5)

    blur_score = float(np.clip(laplacian_log_variance / 0.55, 0.0, 1.0))
    contrast_score = float(np.clip(contrast / 0.20, 0.0, 1.0))
    score = 0.5 * blur_score + 0.2 * contrast_score + 0.3 * exposure
    return {
        "score": score,
        "laplacian_log_variance": laplacian_log_variance,
        "contrast": contrast,
        "exposure": exposure,
    }


def score_smoothness(states: np.ndarray, timestamps: np.ndarray, k: float = 1000.0) -> float:
    if len(states) < 3:
        return 1.0
    accelerations = np.diff(states, n=2, axis=0) / np.diff(timestamps)[:-1, None] ** 2
    rms = float(np.sqrt(np.mean(np.square(accelerations))))
    return float(np.exp(-rms / k))


def score_path_efficiency(states: np.ndarray) -> float:
    path = float(np.sum(np.linalg.norm(np.diff(states, axis=0), axis=1)))
    straight = float(np.linalg.norm(states[-1] - states[0]))
    if path < 1e-6:
        return 0.0
    return float(np.clip(straight / path, 0.0, 1.0))


def score_collision(states: np.ndarray, timestamps: np.ndarray) -> float:
    if len(states) < 3:
        return 1.0
    accelerations = np.diff(states, n=2, axis=0) / np.diff(timestamps)[:-1, None] ** 2
    threshold = 15.0 * np.median(np.abs(accelerations), axis=0, keepdims=True)
    spike_ratio = float(np.mean(np.any(np.abs(accelerations) > threshold, axis=1)))
    return max(0.0, 1.0 - spike_ratio)


def score_joint_stability(states: np.ndarray, timestamps: np.ndarray) -> float:
    final_window = timestamps >= timestamps[-1] - 2.0
    final_state = states[final_window]
    if len(final_state) == 0:
        return 1.0
    joint_std = float(np.std(final_state, axis=0).mean())
    return float(np.exp(-joint_std / 0.05))


def score_actuator_saturation(
    states: np.ndarray,
    actions: np.ndarray,
    threshold_deg: float = 7.0,
) -> float:
    action_state_diff = np.abs(actions[:-1] - states[1:])
    saturation_ratio = float(np.mean(np.any(action_state_diff > threshold_deg, axis=1)))
    return float(np.exp(-4.0 * saturation_ratio))


def score_runtime(runtime_seconds: float, nominal_runtime_seconds: float) -> float:
    if nominal_runtime_seconds <= 0:
        return 1.0
    if runtime_seconds <= nominal_runtime_seconds:
        return 1.0
    overflow_ratio = (runtime_seconds - nominal_runtime_seconds) / nominal_runtime_seconds
    return float(np.exp(-overflow_ratio))
