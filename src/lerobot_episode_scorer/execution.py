"""Vision-language episode execution scoring via stitched frame grid."""

import base64
import io
import json
import multiprocessing as mp
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
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 60.0


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


class ExecutionBackendError(RuntimeError):
    pass


class ExecutionBackendTimeoutError(TimeoutError):
    pass


def _score_with_ollama_request(
    model: str,
    host: str,
    prompt: str,
    image_bytes: bytes,
    think: bool,
    keep_alive: float,
) -> dict:
    client = ollama.Client(host=host)
    response = client.generate(
        model=model,
        prompt=prompt,
        images=[image_bytes],
        think=think,
        keep_alive=keep_alive,
    )
    raw_response = response.get("response", "")
    reasoning_trace = response.get("thinking")
    probability, _ = parse_success_response(raw_response)
    return {
        "score": probability,
        "probability": probability,
        "raw_response": raw_response,
        "reasoning_trace": reasoning_trace,
    }


def _score_with_lmstudio_request(
    model: str,
    base_url: str,
    prompt: str,
    image_bytes: bytes,
    think: bool,
    max_tokens: int,
    timeout_seconds: float,
) -> dict:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response_schema = {
        "type": "object",
        "properties": {
            "success": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["success"],
        "additionalProperties": False,
    }
    if think:
        response_schema["properties"]["reasoning"] = {"type": "string"}
        response_schema["required"] = ["reasoning", "success"]

    request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(
            {
                "model": model,
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
                            {"type": "text", "text": prompt},
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
                "max_tokens": max_tokens,
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
    with urlopen(request, timeout=timeout_seconds) as response:
        response_json = json.loads(response.read().decode("utf-8"))

    message = response_json["choices"][0]["message"]
    content = message.get("content", "")
    parsed_content = json.loads(content)
    raw_response = str(parsed_content["success"])
    reasoning_trace = None
    if think:
        reasoning_trace = str(parsed_content["reasoning"])
    probability, _ = parse_success_response(raw_response)
    return {
        "score": probability,
        "probability": probability,
        "raw_response": raw_response,
        "reasoning_trace": reasoning_trace,
    }


def _ollama_worker_loop(
    model: str,
    host: str,
    think: bool,
    keep_alive: float,
    connection,
) -> None:
    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break

            if request is None:
                break

            prompt = request["prompt"]
            image_bytes = request["image_bytes"]
            try:
                response = {
                    "ok": True,
                    "value": _score_with_ollama_request(
                        model=model,
                        host=host,
                        prompt=prompt,
                        image_bytes=image_bytes,
                        think=think,
                        keep_alive=keep_alive,
                    ),
                }
            except (KeyError, OSError, TimeoutError, ValueError, ollama.ResponseError) as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            try:
                connection.send(response)
            except (BrokenPipeError, EOFError, OSError):
                break
    finally:
        connection.close()


def _lmstudio_worker_loop(
    model: str,
    base_url: str,
    think: bool,
    max_tokens: int,
    timeout_seconds: float,
    connection,
) -> None:
    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break

            if request is None:
                break

            prompt = request["prompt"]
            image_bytes = request["image_bytes"]
            try:
                response = {
                    "ok": True,
                    "value": _score_with_lmstudio_request(
                        model=model,
                        base_url=base_url,
                        prompt=prompt,
                        image_bytes=image_bytes,
                        think=think,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                    ),
                }
            except (json.JSONDecodeError, KeyError, OSError, TimeoutError, ValueError) as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            try:
                connection.send(response)
            except (BrokenPipeError, EOFError, OSError):
                break
    finally:
        connection.close()


class BaseVLMScorer:
    def __init__(
        self,
        border_size: int = DEFAULT_BORDER_SIZE,
        max_image_side: int | None = DEFAULT_MAX_IMAGE_SIDE,
        timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        self.border_size = border_size
        self.max_image_side = max_image_side
        self.timeout_seconds = timeout_seconds
        self._ctx = mp.get_context("spawn")
        self._worker_process = None
        self._worker_connection = None
        self._worker_target = None
        self._worker_args: tuple = ()
        self._backend_name = "backend"

    def _configure_worker(self, backend_name: str, worker_target, worker_args: tuple) -> None:
        self._backend_name = backend_name
        self._worker_target = worker_target
        self._worker_args = worker_args

    def warmup(self) -> None:
        self._ensure_worker()

    def close(self) -> None:
        self._stop_worker(graceful=True)

    def _ensure_worker(self) -> None:
        if self._worker_process is not None and self._worker_process.is_alive():
            return
        if self._worker_target is None:
            raise ExecutionBackendError(f"{self._backend_name} worker is not configured")

        parent_connection, child_connection = self._ctx.Pipe()
        process = self._ctx.Process(
            target=self._worker_target,
            args=(*self._worker_args, child_connection),
        )
        process.start()
        child_connection.close()
        self._worker_connection = parent_connection
        self._worker_process = process

    def _stop_worker(self, graceful: bool) -> None:
        process = self._worker_process
        connection = self._worker_connection
        self._worker_process = None
        self._worker_connection = None

        if process is None:
            if connection is not None:
                connection.close()
            return

        if graceful and connection is not None and process.is_alive():
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass

        if connection is not None:
            connection.close()

        process.join(1 if graceful else 0)
        if process.is_alive():
            process.terminate()
            process.join(5)

    def _call_worker(self, request: dict) -> dict:
        self._ensure_worker()
        if self._worker_connection is None:
            raise ExecutionBackendError(f"{self._backend_name} worker is unavailable")

        try:
            self._worker_connection.send(request)
            has_result = self._worker_connection.poll(self.timeout_seconds)
        except KeyboardInterrupt:
            self._stop_worker(graceful=False)
            raise
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._stop_worker(graceful=False)
            raise ExecutionBackendError(f"{self._backend_name} worker failed: {exc}") from exc

        if not has_result:
            self._stop_worker(graceful=False)
            raise ExecutionBackendTimeoutError(
                f"{self._backend_name} request exceeded {self.timeout_seconds:.1f}s"
            )

        try:
            result = self._worker_connection.recv()
        except KeyboardInterrupt:
            self._stop_worker(graceful=False)
            raise
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._stop_worker(graceful=False)
            raise ExecutionBackendError(f"{self._backend_name} worker failed: {exc}") from exc

        if not result["ok"]:
            raise ExecutionBackendError(f"{self._backend_name} request failed: {result['error']}")
        return result["value"]

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
        timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            border_size=border_size,
            max_image_side=max_image_side,
            timeout_seconds=timeout_seconds,
        )
        self.model = model
        self.host = host
        self.think = think
        self.keep_alive = keep_alive
        self._configure_worker(
            backend_name="Ollama",
            worker_target=_ollama_worker_loop,
            worker_args=(self.model, self.host, self.think, self.keep_alive),
        )

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
        result = self._call_worker({"prompt": prompt, "image_bytes": image_bytes})

        return {
            **result,
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
        timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            border_size=border_size,
            max_image_side=max_image_side,
            timeout_seconds=timeout_seconds,
        )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.think = think
        self.max_tokens = max_tokens
        self._configure_worker(
            backend_name="LM Studio",
            worker_target=_lmstudio_worker_loop,
            worker_args=(
                self.model,
                self.base_url,
                self.think,
                self.max_tokens,
                self.timeout_seconds,
            ),
        )

    def score_episode(
        self,
        episode: EpisodeRecord,
        camera_key: str | None = None,
        pre_extracted: dict[str, Image.Image] | None = None,
    ) -> dict:
        selected_camera = self._select_camera(episode, camera_key)
        stitched_image = self._get_stitched_image(episode, selected_camera, pre_extracted)
        image_bytes = self._encode_image(stitched_image)
        result = self._call_worker(
            {
                "prompt": self._build_prompt(episode.task),
                "image_bytes": image_bytes,
            }
        )

        return {
            **result,
            "camera_used": selected_camera,
        }
