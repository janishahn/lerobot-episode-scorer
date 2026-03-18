from fractions import Fraction
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from lerobot_episode_scorer.video import VideoSegment, sample_episode_frames, sample_segment_frames


class FakeFrame:
    def __init__(self, time: float, value: int) -> None:
        self.time = time
        self.value = value

    def to_ndarray(self, format: str) -> np.ndarray:
        return np.full((1, 1, 3), self.value, dtype=np.uint8)


class FakeStream:
    def __init__(self) -> None:
        self.time_base = Fraction(1, 30)
        self.thread_type: str | None = None


class FakeContainer:
    def __init__(self, frames: list[FakeFrame]) -> None:
        self.frames = frames
        self.stream = FakeStream()
        self.streams = type("Streams", (), {"video": [self.stream]})()
        self.seek_calls: list[tuple[int, FakeStream, bool]] = []

    def seek(self, offset: int, stream: FakeStream, backward: bool) -> None:
        self.seek_calls.append((offset, stream, backward))

    def decode(self, stream: FakeStream):
        self.decode_stream = stream
        return iter(self.frames)

    def __enter__(self) -> "FakeContainer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class VideoSamplingTests(TestCase):
    def test_sample_episode_frames_seeks_to_segment_start(self) -> None:
        container = FakeContainer(
            [
                FakeFrame(9.8, 1),
                FakeFrame(10.0, 2),
                FakeFrame(13.4, 3),
                FakeFrame(16.8, 4),
                FakeFrame(20.0, 5),
            ]
        )
        segment = VideoSegment(Path("/tmp/test.mp4"), from_timestamp=10.0, to_timestamp=20.0)

        with patch("lerobot_episode_scorer.video.av.open", return_value=container):
            frames = sample_episode_frames(segment)

        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [2, 3, 4, 5])
        self.assertEqual(container.stream.thread_type, "AUTO")
        self.assertEqual(container.seek_calls, [(300, container.stream, True)])

    def test_sample_segment_frames_seeks_before_sampling(self) -> None:
        container = FakeContainer(
            [
                FakeFrame(12.0, 1),
                FakeFrame(12.6, 2),
                FakeFrame(15.1, 3),
                FakeFrame(17.6, 4),
            ]
        )
        segment = VideoSegment(Path("/tmp/test.mp4"), from_timestamp=10.0, to_timestamp=20.0)

        with patch("lerobot_episode_scorer.video.av.open", return_value=container):
            frames = sample_segment_frames(segment, num_samples=3)

        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [2, 3, 4])
        self.assertEqual(container.seek_calls, [(300, container.stream, True)])
