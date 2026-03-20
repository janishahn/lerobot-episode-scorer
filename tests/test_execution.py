import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from lerobot_episode_scorer.execution import (
    DEFAULT_BORDER_SIZE,
    DEFAULT_FRAMES_PER_EPISODE,
    DEFAULT_GEMINI_API_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_LMSTUDIO_MODEL,
    DEFAULT_MAX_IMAGE_SIDE,
    GeminiVLMScorer,
    LMStudioVLMScorer,
    OllamaVLMScorer,
    _score_with_gemini_request,
    _score_with_lmstudio_request,
    _score_with_ollama_request,
    parse_success_response,
    stitch_frames,
)
from lerobot_episode_scorer.output import RollingOutputWriter, compute_summary, flatten_episode_row
from lerobot_episode_scorer.quality import score_visual_frame


class TestStitchFrames(unittest.TestCase):
    def test_stitch_four_frames_creates_2x2_grid(self) -> None:
        frames = [np.zeros((100, 80, 3), dtype=np.uint8) for _ in range(4)]
        for i, frame in enumerate(frames):
            frame[:, :, 0] = i * 50
            frame[:, :, 1] = i * 25
            frame[:, :, 2] = 255 - i * 50

        grid = stitch_frames(frames, border_size=4)

        self.assertEqual(grid.size, (2 * 80 + 4, 2 * 100 + 4))
        self.assertEqual(grid.mode, "RGB")

    def test_stitch_frames_requires_exactly_four_frames(self) -> None:
        with self.assertRaises(ValueError) as context:
            stitch_frames([np.zeros((100, 80, 3), dtype=np.uint8) for _ in range(3)])
        self.assertIn("Expected 4 frames", str(context.exception))

    def test_stitch_frames_with_custom_border(self) -> None:
        frames = [np.zeros((100, 80, 3), dtype=np.uint8) for _ in range(4)]
        grid = stitch_frames(frames, border_size=10, border_color=(255, 0, 0))

        border_pixel = grid.getpixel((80, 0))
        self.assertEqual(border_pixel, (255, 0, 0))


class TestParseSuccessResponse(unittest.TestCase):
    def test_parse_yes_as_success(self) -> None:
        probability, raw = parse_success_response("Yes, the task was completed successfully.")
        self.assertEqual(probability, 1.0)
        self.assertEqual(raw, "Yes, the task was completed successfully.")

    def test_parse_no_as_failure(self) -> None:
        probability, raw = parse_success_response("No, the robot failed to grasp the object.")
        self.assertEqual(probability, 0.0)
        self.assertEqual(raw, "No, the robot failed to grasp the object.")

    def test_parse_success_keyword(self) -> None:
        probability, _ = parse_success_response("SUCCESS")
        self.assertEqual(probability, 1.0)

    def test_parse_fail_keyword(self) -> None:
        probability, _ = parse_success_response("FAILURE")
        self.assertEqual(probability, 0.0)

    def test_parse_case_insensitive(self) -> None:
        probability_yes, _ = parse_success_response("YES")
        probability_no, _ = parse_success_response("NO")
        self.assertEqual(probability_yes, 1.0)
        self.assertEqual(probability_no, 0.0)

    def test_parse_unclear_as_0_5(self) -> None:
        probability, _ = parse_success_response("The robot completed most of the task.")
        self.assertEqual(probability, 0.5)

    def test_parse_not_successful_as_failure(self) -> None:
        probability, _ = parse_success_response("The task was not successful.")
        self.assertEqual(probability, 0.0)

    def test_parse_unsuccessful_as_failure(self) -> None:
        probability, _ = parse_success_response("The execution was unsuccessful.")
        self.assertEqual(probability, 0.0)


class TestLMStudioVLMScorer(unittest.TestCase):
    def test_default_constants(self) -> None:
        self.assertEqual(DEFAULT_LMSTUDIO_BASE_URL, "http://localhost:1234/v1")
        self.assertEqual(DEFAULT_BORDER_SIZE, 4)
        self.assertEqual(DEFAULT_FRAMES_PER_EPISODE, 4)
        self.assertEqual(DEFAULT_MAX_IMAGE_SIDE, 448)

    def test_scorer_disables_reasoning_by_default(self) -> None:
        scorer = LMStudioVLMScorer()
        self.assertFalse(scorer.think)

    @patch.object(LMStudioVLMScorer, "_call_worker")
    def test_score_episode_calls_lmstudio_chat_completions(
        self, mock_call_worker: MagicMock
    ) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": None,
        }
        episode = MagicMock()
        episode.task = "pick and place the object"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=10.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
            scorer = LMStudioVLMScorer(
                model=DEFAULT_LMSTUDIO_MODEL,
                base_url=DEFAULT_LMSTUDIO_BASE_URL,
                border_size=DEFAULT_BORDER_SIZE,
            )
            result = scorer.score_episode(episode)

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["probability"], 1.0)
        self.assertEqual(result["camera_used"], "observation.images.top")
        self.assertEqual(result["raw_response"], "yes")
        self.assertIsNone(result["reasoning_trace"])
        request = mock_call_worker.call_args.args[0]
        self.assertIn("pick and place the object", request["prompt"])
        self.assertIsInstance(request["image_bytes"], bytes)

    @patch.object(LMStudioVLMScorer, "_call_worker")
    def test_score_episode_with_think_keeps_reasoning_separate(
        self, mock_call_worker: MagicMock
    ) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": "The cube ends inside the container.",
        }

        episode = MagicMock()
        episode.task = "pick and place the object"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=10.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
            scorer = LMStudioVLMScorer(think=True)
            result = scorer.score_episode(episode)

        self.assertEqual(result["raw_response"], "yes")
        self.assertEqual(result["reasoning_trace"], "The cube ends inside the container.")
        self.assertIn("pick and place the object", mock_call_worker.call_args.args[0]["prompt"])

    @patch.object(LMStudioVLMScorer, "_call_worker")
    def test_score_episode_downscales_image_before_request(
        self, mock_call_worker: MagicMock
    ) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": None,
        }

        episode = MagicMock()
        episode.task = "test task"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=5.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((256, 448, 3), dtype=np.uint8) for _ in range(4)]
            scorer = LMStudioVLMScorer(max_image_side=448)
            scorer.score_episode(episode)

        image = Image.open(io.BytesIO(mock_call_worker.call_args.args[0]["image_bytes"]))
        self.assertEqual(image.size, (448, 257))

    @patch.object(LMStudioVLMScorer, "_call_worker")
    def test_score_episode_uses_task_override_when_provided(
        self, mock_call_worker: MagicMock
    ) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": None,
        }
        episode = MagicMock()
        episode.task = "dataset task"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=10.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
            scorer = LMStudioVLMScorer()
            scorer.score_episode(episode, task="manual override")

        prompt = mock_call_worker.call_args.args[0]["prompt"]
        self.assertIn("manual override", prompt)
        self.assertNotIn("dataset task", prompt)

    @patch("lerobot_episode_scorer.execution.urlopen")
    def test_lmstudio_worker_with_think_requires_reasoning(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"reasoning": "The cube ends inside the container.", '
                                '"success": "yes"}'
                            ),
                        }
                    }
                ]
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response
        result = _score_with_lmstudio_request(
            DEFAULT_LMSTUDIO_MODEL,
            DEFAULT_LMSTUDIO_BASE_URL,
            "Pick and place the object.",
            b"image-bytes",
            True,
            128,
            5.0,
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            payload["response_format"]["json_schema"]["schema"]["required"],
            ["reasoning", "success"],
        )
        self.assertEqual(result["reasoning_trace"], "The cube ends inside the container.")

    @patch("lerobot_episode_scorer.execution.urlopen")
    def test_lmstudio_http_error_includes_response_body(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = HTTPError(
            url="http://localhost:1234/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"insufficient system resources"}}'),
        )

        with self.assertRaises(ValueError) as context:
            _score_with_lmstudio_request(
                DEFAULT_LMSTUDIO_MODEL,
                DEFAULT_LMSTUDIO_BASE_URL,
                "Pick and place the object.",
                b"image-bytes",
                False,
                128,
                5.0,
            )

        self.assertIn("HTTP Error 400: Bad Request", str(context.exception))
        self.assertIn("insufficient system resources", str(context.exception))


class TestGeminiVLMScorer(unittest.TestCase):
    def test_default_constants(self) -> None:
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-flash-latest")
        self.assertEqual(
            DEFAULT_GEMINI_API_URL, "https://generativelanguage.googleapis.com/v1beta/models"
        )

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    def test_scorer_disables_reasoning_by_default(self) -> None:
        scorer = GeminiVLMScorer()
        self.assertFalse(scorer.think)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch.object(GeminiVLMScorer, "_call_worker")
    def test_score_episode_calls_gemini_generate_content(self, mock_call_worker: MagicMock) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": None,
        }
        episode = MagicMock()
        episode.task = "pick and place the object"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=10.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
            scorer = GeminiVLMScorer()
            result = scorer.score_episode(episode)

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["probability"], 1.0)
        self.assertEqual(result["camera_used"], "observation.images.top")
        self.assertEqual(result["raw_response"], "yes")
        request = mock_call_worker.call_args.args[0]
        self.assertIn("pick and place the object", request["prompt"])
        self.assertIsInstance(request["image_bytes"], bytes)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch.object(GeminiVLMScorer, "_call_worker")
    def test_score_episode_with_think_keeps_reasoning_separate(
        self, mock_call_worker: MagicMock
    ) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": "The object ends in the tray.",
        }
        episode = MagicMock()
        episode.task = "pick and place the object"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=10.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
            scorer = GeminiVLMScorer(think=True)
            result = scorer.score_episode(episode)

        self.assertEqual(result["reasoning_trace"], "The object ends in the tray.")
        self.assertIn("pick and place the object", mock_call_worker.call_args.args[0]["prompt"])

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch.object(GeminiVLMScorer, "_call_worker")
    def test_score_episode_downscales_image_before_request(
        self, mock_call_worker: MagicMock
    ) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": None,
        }
        episode = MagicMock()
        episode.task = "test task"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=5.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((256, 448, 3), dtype=np.uint8) for _ in range(4)]
            scorer = GeminiVLMScorer(max_image_side=448)
            scorer.score_episode(episode)

        image = Image.open(io.BytesIO(mock_call_worker.call_args.args[0]["image_bytes"]))
        self.assertEqual(image.size, (448, 257))

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch.object(GeminiVLMScorer, "_call_worker")
    def test_score_episode_uses_task_override_when_provided(
        self, mock_call_worker: MagicMock
    ) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": None,
        }
        episode = MagicMock()
        episode.task = "dataset task"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=10.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
            scorer = GeminiVLMScorer()
            scorer.score_episode(episode, task="manual override")

        prompt = mock_call_worker.call_args.args[0]["prompt"]
        self.assertIn("manual override", prompt)
        self.assertNotIn("dataset task", prompt)

    def test_missing_api_key_raises_value_error(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as context:
                GeminiVLMScorer()
        self.assertIn("GEMINI_API_KEY", str(context.exception))

    @patch("lerobot_episode_scorer.execution.urlopen")
    def test_gemini_request_uses_structured_output_and_minimal_thinking(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"success":"yes"}',
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = _score_with_gemini_request(
            DEFAULT_GEMINI_MODEL,
            DEFAULT_GEMINI_API_URL,
            "test-key",
            "Pick and place the object.",
            b"image-bytes",
            False,
            5.0,
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            request.full_url, f"{DEFAULT_GEMINI_API_URL}/{DEFAULT_GEMINI_MODEL}:generateContent"
        )
        self.assertEqual(request.headers["X-goog-api-key"], "test-key")
        self.assertEqual(
            payload["contents"][0]["parts"][0]["inline_data"]["mime_type"],
            "image/jpeg",
        )
        self.assertEqual(payload["contents"][0]["parts"][1]["text"], "Pick and place the object.")
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(
            payload["generationConfig"]["responseJsonSchema"]["properties"]["success"]["enum"],
            ["yes", "no"],
        )
        self.assertEqual(
            payload["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "minimal",
        )
        self.assertEqual(result["raw_response"], "yes")
        self.assertIsNone(result["reasoning_trace"])

    @patch("lerobot_episode_scorer.execution.urlopen")
    def test_gemini_request_with_think_collects_thought_parts(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "First thought.", "thought": True},
                                {"text": "Second thought.", "thought": True},
                                {"text": '{"success":"yes"}'},
                            ]
                        }
                    }
                ]
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = _score_with_gemini_request(
            DEFAULT_GEMINI_MODEL,
            DEFAULT_GEMINI_API_URL,
            "test-key",
            "prompt",
            b"image-bytes",
            True,
            5.0,
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["generationConfig"]["thinkingConfig"]["thinkingLevel"], "low")
        self.assertTrue(payload["generationConfig"]["thinkingConfig"]["includeThoughts"])
        self.assertEqual(result["reasoning_trace"], "First thought.\nSecond thought.")

    @patch("lerobot_episode_scorer.execution.urlopen")
    def test_gemini_request_requires_candidates(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"candidates": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            _score_with_gemini_request(
                DEFAULT_GEMINI_MODEL,
                DEFAULT_GEMINI_API_URL,
                "test-key",
                "prompt",
                b"image-bytes",
                False,
                5.0,
            )

        self.assertIn("did not include candidates", str(context.exception))

    @patch("lerobot_episode_scorer.execution.urlopen")
    def test_gemini_http_error_includes_response_body(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = HTTPError(
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"quota exceeded"}}'),
        )

        with self.assertRaises(ValueError) as context:
            _score_with_gemini_request(
                DEFAULT_GEMINI_MODEL,
                DEFAULT_GEMINI_API_URL,
                "test-key",
                "prompt",
                b"image-bytes",
                False,
                5.0,
            )

        self.assertIn("HTTP Error 429: Too Many Requests", str(context.exception))
        self.assertIn("quota exceeded", str(context.exception))


class TestOllamaVLMScorer(unittest.TestCase):
    def test_scorer_disables_thinking_by_default(self) -> None:
        scorer = OllamaVLMScorer()

        self.assertFalse(scorer.think)

    @patch.object(OllamaVLMScorer, "_call_worker")
    def test_score_episode_calls_ollama_generate(self, mock_call_worker: MagicMock) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "Yes, the task was successful.",
            "reasoning_trace": None,
        }
        episode = MagicMock()
        episode.task = "pick and place the object"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=10.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
            scorer = OllamaVLMScorer(border_size=DEFAULT_BORDER_SIZE)
            result = scorer.score_episode(episode)

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["probability"], 1.0)
        self.assertEqual(result["camera_used"], "observation.images.top")
        self.assertEqual(result["raw_response"], "Yes, the task was successful.")
        self.assertIn("pick and place the object", mock_call_worker.call_args.args[0]["prompt"])

    @patch.object(OllamaVLMScorer, "_call_worker")
    def test_score_episode_captures_reasoning_trace(self, mock_call_worker: MagicMock) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "Yes, successful.",
            "reasoning_trace": "Let me analyze each frame...",
        }

        episode = MagicMock()
        episode.task = "test task"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=5.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
            scorer = OllamaVLMScorer(think=True)
            result = scorer.score_episode(episode)

        self.assertEqual(result["reasoning_trace"], "Let me analyze each frame...")

    @patch.object(OllamaVLMScorer, "_call_worker")
    def test_score_episode_keeps_image_size_when_downscaling_disabled(
        self, mock_call_worker: MagicMock
    ) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": None,
        }

        episode = MagicMock()
        episode.task = "test task"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=5.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((256, 448, 3), dtype=np.uint8) for _ in range(4)]
            scorer = OllamaVLMScorer(max_image_side=None)
            scorer.score_episode(episode)

        image_bytes = mock_call_worker.call_args.args[0]["image_bytes"]
        image = Image.open(io.BytesIO(image_bytes))
        self.assertEqual(image.size, (900, 516))

    @patch.object(OllamaVLMScorer, "_call_worker")
    def test_score_episode_uses_task_override_when_provided(
        self, mock_call_worker: MagicMock
    ) -> None:
        mock_call_worker.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": None,
        }
        episode = MagicMock()
        episode.task = "dataset task"
        episode.cameras = {
            "observation.images.top": MagicMock(
                video_path="/tmp/test.mp4",
                from_timestamp=0.0,
                to_timestamp=5.0,
            )
        }

        with patch("lerobot_episode_scorer.execution.sample_episode_frames") as mock_sample:
            mock_sample.return_value = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
            scorer = OllamaVLMScorer()
            scorer.score_episode(episode, task="manual override")

        prompt = mock_call_worker.call_args.args[0]["prompt"]
        self.assertIn("manual override", prompt)
        self.assertNotIn("dataset task", prompt)

    @patch("lerobot_episode_scorer.execution.ollama.Client")
    def test_ollama_worker_captures_reasoning_trace(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": "Yes, successful.",
            "thinking": "Let me analyze each frame...",
        }
        mock_client_class.return_value = mock_client
        result = _score_with_ollama_request(
            "model",
            "http://localhost:11434",
            "prompt",
            b"image-bytes",
            True,
            300.0,
        )

        mock_client.generate.assert_called_once_with(
            model="model",
            prompt="prompt",
            images=[b"image-bytes"],
            think=True,
            keep_alive=300.0,
        )
        self.assertEqual(result["reasoning_trace"], "Let me analyze each frame...")


class TestComputeSummary(unittest.TestCase):
    def test_flatten_episode_row_uses_stable_camera_columns(self) -> None:
        row = {
            "repo_id": "demo/repo",
            "dataset_family": "test",
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
            "execution_backend": "lmstudio",
            "vlm_response": "yes",
            "reasoning_trace": None,
            "camera_used": "observation.images.top",
        }

        flat_row = flatten_episode_row(
            row,
            ["observation.images.top", "observation.images.wrist"],
        )

        self.assertEqual(flat_row["observation_images_top_visual"], 0.7)
        self.assertIsNone(flat_row["observation_images_wrist_visual"])

    def test_rolling_output_writer_keeps_csv_schema_across_rows(self) -> None:
        base_row = {
            "repo_id": "demo/repo",
            "dataset_family": "test",
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
            "execution_backend": "lmstudio",
            "vlm_response": "yes",
            "reasoning_trace": None,
            "camera_used": "observation.images.top",
        }

        with tempfile.TemporaryDirectory() as tempdir:
            writer = RollingOutputWriter(
                output_dir=Path(tempdir),
                execution_backend="lmstudio",
                nominal_runtime_seconds=10.0,
                camera_keys=["observation.images.top", "observation.images.wrist"],
                execution_model="qwen/qwen3.5-9b",
                repo_id="demo/repo",
                dataset_family="test",
            )
            writer.add_episode(
                {
                    **base_row,
                    "episode_index": 0,
                }
            )
            writer.add_episode(
                {
                    **base_row,
                    "episode_index": 1,
                    "quality_components": {
                        **base_row["quality_components"],
                        "visual_clarity_by_camera": {
                            "observation.images.wrist": 0.6,
                            "observation.images.top": 0.5,
                        },
                    },
                }
            )
            writer.finalize()

            with (Path(tempdir) / "episode_scores.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["observation_images_top_visual"], "0.7")
        self.assertEqual(rows[0]["observation_images_wrist_visual"], "")
        self.assertEqual(rows[1]["observation_images_top_visual"], "0.5")
        self.assertEqual(rows[1]["observation_images_wrist_visual"], "0.6")

    def test_quality_and_execution_metrics_included(self) -> None:
        rows = [
            {
                "repo_id": "demo/repo",
                "dataset_family": "test",
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
                "execution_backend": "lmstudio",
                "vlm_response": "yes",
                "camera_used": "observation.images.top",
            },
            {
                "repo_id": "demo/repo",
                "dataset_family": "test",
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
                "execution_backend": "lmstudio",
                "vlm_response": "no",
                "camera_used": "observation.images.top",
            },
        ]

        summary = compute_summary(
            rows=rows,
            execution_backend="lmstudio",
            nominal_runtime_seconds=10.0,
            camera_keys=["observation.images.top"],
            execution_model="qwen/qwen3.5-9b",
            repo_id="demo/repo",
            dataset_family="test",
        )

        self.assertEqual(summary["labels_available"], 2)
        self.assertIn("quality_metrics", summary)
        self.assertIn("execution_metrics", summary)
        self.assertIn("combined_metrics", summary)
        self.assertEqual(summary["execution_model"], "qwen/qwen3.5-9b")

    def test_visual_score_responds_to_detail_and_exposure(self) -> None:
        flat_bright = np.full((16, 16, 3), 255, dtype=np.uint8)
        textured = np.zeros((16, 16, 3), dtype=np.uint8)
        textured[:, :8] = 40
        textured[:, 8:] = 200

        flat_score = score_visual_frame(flat_bright)["score"]
        textured_score = score_visual_frame(textured)["score"]
        self.assertLess(flat_score, textured_score)


if __name__ == "__main__":
    unittest.main()
