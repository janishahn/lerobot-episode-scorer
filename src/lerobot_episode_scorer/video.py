from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class VideoSegment:
    video_path: Path
    from_timestamp: float
    to_timestamp: float


def sample_segment_frames(segment: VideoSegment, num_samples: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    target_times = np.linspace(
        segment.from_timestamp,
        segment.to_timestamp,
        num=num_samples + 2,
        endpoint=True,
    )[1:-1]

    with av.open(str(segment.video_path)) as container:
        stream = container.streams.video[0]
        target_index = 0

        for frame in container.decode(stream):
            frame_time = float(frame.time)
            if frame_time < target_times[target_index]:
                continue

            frames.append(frame.to_ndarray(format="rgb24"))
            target_index += 1
            if target_index == len(target_times):
                break

    if len(frames) != len(target_times):
        raise RuntimeError(
            f"Could not sample {len(target_times)} frames from {segment.video_path}; got {len(frames)}"
        )

    return frames
