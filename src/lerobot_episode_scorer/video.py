from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

VALIDATION_DURATION_TOLERANCE_SECONDS = 1e-3


class VideoValidationError(Exception):
    """Raised when a video file fails pre-validation."""

    pass


@dataclass(frozen=True)
class VideoSegment:
    video_path: Path
    from_timestamp: float
    to_timestamp: float


def validate_video(segment: VideoSegment) -> None:
    """Validate that a video segment can be opened and decoded.

    Raises VideoValidationError if the file is missing, empty, corrupted,
    or cannot be opened/decoded.
    """
    path = segment.video_path

    if not path.exists():
        raise VideoValidationError(f"Video file does not exist: {path}")

    if path.stat().st_size == 0:
        raise VideoValidationError(f"Video file is empty: {path}")

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoValidationError(f"No video stream found in: {path}")

            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base) if stream.duration else None
            if (
                duration is not None
                and segment.to_timestamp > duration + VALIDATION_DURATION_TOLERANCE_SECONDS
            ):
                raise VideoValidationError(
                    f"Segment end time {segment.to_timestamp}s exceeds video duration {duration}s: {path}"
                )
    except av.error.InvalidDataError as e:
        raise VideoValidationError(f"Corrupted or invalid video file: {path}") from e
    except av.error.FFmpegError as e:
        raise VideoValidationError(f"Could not open video file with FFmpeg: {path}") from e


def _sample_frames_at_times(
    segment: VideoSegment, target_times: list[float] | np.ndarray
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []

    with av.open(str(segment.video_path)) as container:
        stream = container.streams.video[0]
        start_pts = int(segment.from_timestamp / float(stream.time_base))
        if start_pts > 0:
            container.seek(start_pts, stream=stream, backward=True)

        target_index = 0

        for frame in container.decode(stream):
            frame_time = float(frame.time)
            if frame_time < target_times[target_index]:
                continue

            frames.append(frame.to_ndarray(format="rgb24"))
            target_index += 1
            if target_index >= len(target_times):
                break

    if len(frames) != len(target_times):
        raise RuntimeError(
            f"Could not sample {len(target_times)} frames from {segment.video_path}; got {len(frames)}"
        )

    return frames


def sample_segment_frames(segment: VideoSegment, num_samples: int) -> list[np.ndarray]:
    target_times = np.linspace(
        segment.from_timestamp,
        segment.to_timestamp,
        num=num_samples + 2,
        endpoint=True,
    )[1:-1]
    return _sample_frames_at_times(segment, target_times)


def sample_episode_frames(segment: VideoSegment, num_frames: int = 4) -> list[np.ndarray]:
    """Sample frames at strategic positions throughout the episode.

    For 4 frames: start, ~33%%, ~67%%, end-epsilon.
    For other counts: evenly distributed with emphasis on start/end.
    """
    duration = segment.to_timestamp - segment.from_timestamp

    if num_frames == 4:
        positions = [0.0, 0.33, 0.67, 0.99]
    else:
        positions = [i / (num_frames - 1) * 0.99 for i in range(num_frames)]

    target_times = [segment.from_timestamp + p * duration for p in positions]
    return _sample_frames_at_times(segment, target_times)
