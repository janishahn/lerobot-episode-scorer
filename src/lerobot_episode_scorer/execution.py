"""Vision-language episode execution scoring via stitched frame grid."""

import base64
import io
import json
from urllib.request import Request, urlopen

import ollama
from PIL import Image

from lerobot_episode_scorer.dataset import EpisodeRecord
from lerobot_episode_scorer.video import sample_episode_frames

DEFAULT_OLLAMA_MODEL = "qwen3.5:0.8b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LMSTUDIO_MODEL = "qwen/qwen3.5-9b"
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_BORDER_SIZE = 4
DEFAULT_FRAMES_PER_EPISODE = 4
DEFAULT_MAX_IMAGE_SIDE = 448
DEFAULT_LMSTUDIO_MAX_TOKENS = 128


def stitch_frames(
    frames: list,
    border_size: int = DEFAULT_BORDER_SIZE,
    border_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    if len(frames) != DEFAULT_FRAMES_PER_EPISODE:
        raise ValueError(f"Expected {DEFAULT_FRAMES_PER_EPISODE} frames, got {len(frames)}")
    frame_height, frame_width = frames[0].shape[:2]

    grid_width = 2 * frame_width + border_size
    grid_height = 2 * frame_height + border_size

    grid_image = Image.new("RGB", (grid_width, grid_height), border_color)

    positions = [
        (0, 0),
        (frame_width + border_size, 0),
        (0, frame_height + border_size),
        (frame_width + border_size, frame_height + border_size),
    ]

    for frame, (x, y) in zip(frames, positions, strict=True):
        pil_frame = Image.fromarray(frame)
        grid_image.paste(pil_frame, (x, y))

    return grid_image


def parse_success_response(response: str) -> tuple[float, str]:
    lowered = response.strip().lower()

    if any(neg in lowered for neg in ["not success", "not successful", "unsuccessful"]):
        return 0.0, response

    success_indicators = {"yes", "success", "successful", "pass", "passed", "true", "1"}
    failure_indicators = {"no", "fail", "failure", "failed", "false", "0"}

    for indicator in success_indicators:
        if indicator in lowered:
            return 1.0, response

    for indicator in failure_indicators:
        if indicator in lowered:
            return 0.0, response

    return 0.5, response


class BaseVLMScorer:
    def __init__(
        self,
        border_size: int = DEFAULT_BORDER_SIZE,
        max_image_side: int | None = DEFAULT_MAX_IMAGE_SIDE,
    ) -> None:
        self.border_size = border_size
        self.max_image_side = max_image_side

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        if self.max_image_side is None:
            return image

        width, height = image.size
        largest_side = max(width, height)
        if largest_side <= self.max_image_side:
            return image

        scale = self.max_image_side / largest_side
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        return image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    def _select_camera(self, episode: EpisodeRecord, camera_key: str | None) -> str:
        camera_keys = list(episode.cameras.keys())
        if camera_key:
            if camera_key not in camera_keys:
                raise ValueError(f"Camera key '{camera_key}' not found. Available: {camera_keys}")
            return camera_key

        preferred_order = {"observation.images.top": 0, "observation.images.wrist": 1}
        camera_keys_sorted = sorted(camera_keys, key=lambda key: preferred_order.get(key, 99))
        return camera_keys_sorted[0]

    def _get_stitched_image(
        self,
        episode: EpisodeRecord,
        selected_camera: str,
        pre_extracted: dict[str, Image.Image] | None,
    ) -> Image.Image:
        if pre_extracted and selected_camera in pre_extracted:
            return pre_extracted[selected_camera]

        video_segment = episode.cameras[selected_camera]
        frames = sample_episode_frames(video_segment, DEFAULT_FRAMES_PER_EPISODE)
        stitched_image = stitch_frames(frames, self.border_size)
        return self._prepare_image(stitched_image)

    def _build_prompt(self, task: str) -> str:
        return (
            "The image shows 4 frames from a robot episode arranged in a 2x2 grid "
            "(top-left: start, top-right: ~33% progress, bottom-left: ~67% progress, "
            "bottom-right: end).\n"
            "Does this image show a successful execution of the following task?\n"
            f'Task: "{task}"\n\n'
            "Answer with 'yes' or 'no'."
        )

    def _encode_image(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    def pre_extract_frames_for_episode(self, episode: EpisodeRecord) -> dict[str, Image.Image]:
        pre_extracted = {}
        for camera_key, video_segment in episode.cameras.items():
            frames = sample_episode_frames(video_segment, DEFAULT_FRAMES_PER_EPISODE)
            stitched_image = stitch_frames(frames, self.border_size)
            pre_extracted[camera_key] = self._prepare_image(stitched_image)
        return pre_extracted


class OllamaVLMScorer(BaseVLMScorer):
    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        border_size: int = DEFAULT_BORDER_SIZE,
        think: bool = False,
        keep_alive: float = 5 * 60.0,
        max_image_side: int | None = DEFAULT_MAX_IMAGE_SIDE,
    ) -> None:
        super().__init__(border_size=border_size, max_image_side=max_image_side)
        self.model = model
        self.host = host
        self.think = think
        self.keep_alive = keep_alive
        self._client = ollama.Client(host=host)

    def score_episode(
        self,
        episode: EpisodeRecord,
        camera_key: str | None = None,
        pre_extracted: dict[str, Image.Image] | None = None,
    ) -> dict:
        selected_camera = self._select_camera(episode, camera_key)
        stitched_image = self._get_stitched_image(episode, selected_camera, pre_extracted)
        prompt = self._build_prompt(episode.task)
        image_bytes = self._encode_image(stitched_image)

        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            images=[image_bytes],
            think=self.think,
            keep_alive=self.keep_alive,
        )

        raw_response = response.get("response", "")
        reasoning_trace = response.get("thinking")
        probability, _ = parse_success_response(raw_response)

        return {
            "score": probability,
            "probability": probability,
            "raw_response": raw_response,
            "reasoning_trace": reasoning_trace,
            "camera_used": selected_camera,
        }


class LMStudioVLMScorer(BaseVLMScorer):
    def __init__(
        self,
        model: str = DEFAULT_LMSTUDIO_MODEL,
        base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
        border_size: int = DEFAULT_BORDER_SIZE,
        think: bool = False,
        max_image_side: int | None = DEFAULT_MAX_IMAGE_SIDE,
        max_tokens: int = DEFAULT_LMSTUDIO_MAX_TOKENS,
    ) -> None:
        super().__init__(border_size=border_size, max_image_side=max_image_side)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.think = think
        self.max_tokens = max_tokens

    def score_episode(
        self,
        episode: EpisodeRecord,
        camera_key: str | None = None,
        pre_extracted: dict[str, Image.Image] | None = None,
    ) -> dict:
        selected_camera = self._select_camera(episode, camera_key)
        stitched_image = self._get_stitched_image(episode, selected_camera, pre_extracted)
        image_bytes = self._encode_image(stitched_image)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        response_schema = {
            "type": "object",
            "properties": {
                "success": {"type": "string", "enum": ["yes", "no"]},
            },
            "required": ["success"],
            "additionalProperties": False,
        }
        if self.think:
            response_schema["properties"]["reasoning"] = {"type": "string"}
            response_schema["required"] = ["reasoning", "success"]

        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Assess the robot episode and return JSON that matches the "
                                "schema exactly."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self._build_prompt(episode.task),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                    },
                                },
                            ],
                        },
                    ],
                    "temperature": 0,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "episode_success",
                            "schema": response_schema,
                        },
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=300) as response:
            response_json = json.loads(response.read().decode("utf-8"))

        message = response_json["choices"][0]["message"]
        content = message.get("content", "")
        parsed_content = json.loads(content)
        raw_response = str(parsed_content["success"])
        reasoning_trace = None
        if self.think:
            reasoning_trace = str(parsed_content["reasoning"])
        probability, _ = parse_success_response(raw_response)

        return {
            "score": probability,
            "probability": probability,
            "raw_response": raw_response,
            "reasoning_trace": reasoning_trace,
            "camera_used": selected_camera,
        }
