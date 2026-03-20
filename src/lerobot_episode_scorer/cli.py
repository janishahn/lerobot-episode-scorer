import argparse
import logging
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from lerobot_episode_scorer.dataset import load_lerobot_dataset
from lerobot_episode_scorer.execution import (
    DEFAULT_BORDER_SIZE,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_LMSTUDIO_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    ExecutionBackendError,
    ExecutionBackendTimeoutError,
    GeminiVLMScorer,
    LMStudioVLMScorer,
    OllamaVLMScorer,
)
from lerobot_episode_scorer.output import RollingOutputWriter
from lerobot_episode_scorer.quality import EpisodeQualityScorer
from lerobot_episode_scorer.video import VideoValidationError, validate_video

SAVE_EPISODE_SAMPLE_COUNT = 20
SAVE_FRAMES_DIR = "saved_frames"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score LeRobot episodes with quality metrics and optional VLM-based execution scoring."
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id.")
    parser.add_argument("--root", type=Path, default=None, help="Optional local LeRobot root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for score outputs.",
    )
    parser.add_argument(
        "--dataset-family",
        default="custom",
        help="Dataset family name stored in the outputs.",
    )
    parser.add_argument(
        "--camera-key",
        action="append",
        default=[],
        help="Camera key to score. Use bare names like 'top' and 'wrist' or full feature keys.",
    )
    parser.add_argument(
        "--nominal-runtime-seconds",
        type=float,
        default=None,
        help="Runtime target used for runtime scoring. Defaults to the dataset median runtime.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Override the dataset task instruction used for VLM execution scoring.",
    )
    parser.add_argument(
        "--execution-backend",
        choices=["none", "gemini", "lmstudio", "ollama"],
        default="gemini",
        help="Execution backend: 'none', 'gemini' (default), 'lmstudio', or 'ollama'.",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_GEMINI_MODEL}).",
    )
    parser.add_argument(
        "--lmstudio-model",
        default=DEFAULT_LMSTUDIO_MODEL,
        help=f"LM Studio model to use (default: {DEFAULT_LMSTUDIO_MODEL}).",
    )
    parser.add_argument(
        "--lmstudio-base-url",
        default=DEFAULT_LMSTUDIO_BASE_URL,
        help=f"LM Studio OpenAI-compatible base URL (default: {DEFAULT_LMSTUDIO_BASE_URL}).",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_OLLAMA_MODEL}).",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama host URL (default: {DEFAULT_OLLAMA_HOST}).",
    )
    parser.add_argument(
        "--stitch-border-size",
        type=int,
        default=DEFAULT_BORDER_SIZE,
        help=f"Border size in pixels for stitched frame grid (default: {DEFAULT_BORDER_SIZE}).",
    )
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument(
        "--think",
        action="store_true",
        dest="think",
        help="Enable reasoning traces. Disabled by default.",
    )
    think_group.add_argument(
        "--no-think",
        action="store_false",
        dest="think",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(think=False)
    parser.add_argument(
        "--ollama-keep-alive",
        type=float,
        default=300.0,
        help="Seconds to keep the Ollama model loaded in memory (default: 300). Set to -1 for infinite.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help=(
            f"Save stitched frame grids for up to {SAVE_EPISODE_SAMPLE_COUNT} random episodes "
            "to disk for debugging."
        ),
    )
    parser.add_argument(
        "--execution-timeout-seconds",
        type=float,
        default=DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        help=(
            "Maximum time to wait for a single Gemini, LM Studio, or Ollama scoring request before "
            "marking that episode's execution score as unknown."
        ),
    )
    return parser


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        filename="scoring.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    args = build_parser().parse_args()

    loaded_dataset = load_lerobot_dataset(args.repo_id, args.root, args.camera_key)
    nominal_runtime_seconds = (
        args.nominal_runtime_seconds
        if args.nominal_runtime_seconds is not None
        else loaded_dataset.nominal_runtime_seconds
    )
    print(
        f"Loaded {len(loaded_dataset.episodes)} episodes from {loaded_dataset.repo_id} "
        f"with cameras {loaded_dataset.camera_keys}"
    )

    quality_scorer = EpisodeQualityScorer(nominal_runtime_seconds=nominal_runtime_seconds)
    execution_scorer = None
    execution_model = ""
    pre_extracted_frames: dict[int, dict[str, Image.Image]] = {}

    if args.execution_backend == "gemini":
        execution_scorer = GeminiVLMScorer(
            model=args.gemini_model,
            border_size=args.stitch_border_size,
            think=args.think,
            timeout_seconds=args.execution_timeout_seconds,
        )
        execution_model = args.gemini_model
        print(f"Using Gemini VLM scorer with model {args.gemini_model}")
    elif args.execution_backend == "lmstudio":
        execution_scorer = LMStudioVLMScorer(
            model=args.lmstudio_model,
            base_url=args.lmstudio_base_url,
            border_size=args.stitch_border_size,
            think=args.think,
            timeout_seconds=args.execution_timeout_seconds,
        )
        execution_model = args.lmstudio_model
        print(f"Using LM Studio VLM scorer with model {args.lmstudio_model}")
    elif args.execution_backend == "ollama":
        execution_scorer = OllamaVLMScorer(
            model=args.ollama_model,
            host=args.ollama_host,
            border_size=args.stitch_border_size,
            think=args.think,
            keep_alive=args.ollama_keep_alive,
            timeout_seconds=args.execution_timeout_seconds,
        )
        execution_model = args.ollama_model
        print(f"Using Ollama VLM scorer with model {args.ollama_model}")

    if execution_scorer is not None:
        execution_scorer.warmup()
        print("Validating video files for all episodes...")
        validation_errors: list[tuple[int, str, str]] = []
        for episode in tqdm(loaded_dataset.episodes, desc="Validating videos", unit="episode"):
            for camera_key, video_segment in episode.cameras.items():
                try:
                    validate_video(video_segment)
                except VideoValidationError as e:
                    validation_errors.append((episode.episode_index, camera_key, str(e)))

        if validation_errors:
            print(f"\nFound {len(validation_errors)} video validation errors:")
            for episode_idx, camera, error in validation_errors[:10]:
                print(f"  Episode {episode_idx} ({camera}): {error}")
            if len(validation_errors) > 10:
                print(f"  ... and {len(validation_errors) - 10} more errors")
            print("\nPlease fix the video files or re-download the dataset before proceeding.")
            return

        print("Pre-extracting frames for all episodes...")
        for episode in tqdm(loaded_dataset.episodes, desc="Extracting frames", unit="episode"):
            start_time = time.time()
            pre_extracted_frames[episode.episode_index] = (
                execution_scorer.pre_extract_frames_for_episode(episode)
            )
            elapsed = time.time() - start_time
            logging.info(f"Frame extraction episode {episode.episode_index}: {elapsed:.2f}s")

        print("Frame extraction complete.")

    if args.save_frames and pre_extracted_frames:
        frames_dir = args.output_dir / SAVE_FRAMES_DIR
        frames_dir.mkdir(parents=True, exist_ok=True)
        sampled_episode_indices = random.sample(
            list(pre_extracted_frames),
            k=min(SAVE_EPISODE_SAMPLE_COUNT, len(pre_extracted_frames)),
        )
        print(
            "Saving stitched frame grids for "
            f"{len(sampled_episode_indices)} sampled episodes to disk..."
        )
        for episode_index in sampled_episode_indices:
            for camera_key, stitched_image in pre_extracted_frames[episode_index].items():
                filepath = (
                    frames_dir / f"episode_{episode_index:03d}_{camera_key.replace('.', '_')}.jpg"
                )
                stitched_image.save(filepath, "JPEG", quality=95)
                logging.info("Saved frame: %s", filepath)

    print("Starting episode scoring...")
    writer = RollingOutputWriter(
        output_dir=args.output_dir,
        execution_backend=args.execution_backend,
        nominal_runtime_seconds=nominal_runtime_seconds,
        camera_keys=loaded_dataset.camera_keys,
        execution_model=execution_model,
        repo_id=loaded_dataset.repo_id,
        dataset_family=args.dataset_family,
    )

    try:
        for episode in tqdm(loaded_dataset.episodes, desc="Scoring episodes", unit="episode"):
            start_time = time.time()
            quality = quality_scorer.score_episode(episode)
            execution_task = args.instruction if args.instruction is not None else episode.task

            if execution_scorer is None:
                execution_score = 1.0
                execution_probability = 1.0
                vlm_response = None
                reasoning_trace = None
                camera_used = loaded_dataset.camera_keys[0] if loaded_dataset.camera_keys else None
            else:
                try:
                    result = execution_scorer.score_episode(
                        episode,
                        task=execution_task,
                        pre_extracted=pre_extracted_frames.get(episode.episode_index),
                    )
                    execution_score = result["score"]
                    execution_probability = result["probability"]
                    vlm_response = result.get("raw_response")
                    reasoning_trace = result.get("reasoning_trace")
                    camera_used = result.get("camera_used")
                except (ExecutionBackendError, ExecutionBackendTimeoutError) as e:
                    execution_score = 0.5
                    execution_probability = 0.5
                    vlm_response = None
                    reasoning_trace = str(e)
                    camera_used = None
                    logging.warning(
                        "Execution scoring failed for episode %s: %s",
                        episode.episode_index,
                        e,
                    )

            elapsed = time.time() - start_time
            logging.info(
                f"Episode {episode.episode_index}: {elapsed:.2f}s - "
                f"quality={quality['aggregate']:.3f}, execution={execution_score:.3f}"
            )

            writer.add_episode(
                {
                    "repo_id": loaded_dataset.repo_id,
                    "dataset_family": args.dataset_family,
                    "episode_index": episode.episode_index,
                    "task": execution_task,
                    "label": episode.label,
                    "quality_score": quality["aggregate"],
                    "execution_score": execution_score,
                    "execution_probability": execution_probability,
                    "combined_score": quality["aggregate"] * execution_score,
                    "runtime_seconds": quality["runtime_seconds"],
                    "quality_components": quality,
                    "execution_backend": args.execution_backend,
                    "vlm_response": vlm_response,
                    "reasoning_trace": reasoning_trace,
                    "camera_used": camera_used,
                }
            )
    except KeyboardInterrupt:
        print("\nInterrupted. Partial outputs were written to disk.")
        raise SystemExit(130) from None
    finally:
        if execution_scorer is not None:
            execution_scorer.close()
        writer.finalize()


if __name__ == "__main__":
    main()
