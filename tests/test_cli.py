from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from lerobot_episode_scorer import cli


class CliInterruptTests(TestCase):
    @patch("lerobot_episode_scorer.cli.validate_video")
    @patch("lerobot_episode_scorer.cli.RollingOutputWriter")
    @patch("lerobot_episode_scorer.cli.GeminiVLMScorer")
    @patch("lerobot_episode_scorer.cli.EpisodeQualityScorer")
    @patch("lerobot_episode_scorer.cli.load_lerobot_dataset")
    @patch("lerobot_episode_scorer.cli.build_parser")
    def test_keyboard_interrupt_is_reraised_after_flushing_outputs(
        self,
        mock_build_parser: MagicMock,
        mock_load_dataset: MagicMock,
        mock_quality_scorer_class: MagicMock,
        mock_gemini_scorer_class: MagicMock,
        mock_writer_class: MagicMock,
        mock_validate_video: MagicMock,
    ) -> None:
        mock_build_parser.return_value.parse_args.return_value = SimpleNamespace(
            repo_id="demo/repo",
            root=None,
            output_dir=Path("/tmp/demo"),
            dataset_family="custom",
            camera_key=[],
            nominal_runtime_seconds=None,
            instruction=None,
            execution_backend="gemini",
            gemini_model="gemini-flash-latest",
            lmstudio_model="demo-lmstudio-model",
            lmstudio_base_url="http://localhost:1234/v1",
            ollama_model="demo-ollama",
            ollama_host="http://localhost:11434",
            stitch_border_size=4,
            think=False,
            ollama_keep_alive=300.0,
            save_frames=False,
            execution_timeout_seconds=10.0,
        )
        episode = SimpleNamespace(
            episode_index=0,
            task="pick",
            label=None,
            cameras={"observation.images.top": MagicMock()},
        )
        mock_load_dataset.return_value = SimpleNamespace(
            repo_id="demo/repo",
            camera_keys=["observation.images.top"],
            nominal_runtime_seconds=1.0,
            episodes=[episode],
        )
        mock_quality_scorer_class.return_value.score_episode.return_value = {"aggregate": 0.5}
        scorer = mock_gemini_scorer_class.return_value
        scorer.pre_extract_frames_for_episode.return_value = {}
        scorer.score_episode.side_effect = KeyboardInterrupt
        writer = mock_writer_class.return_value

        with self.assertRaises(SystemExit) as context:
            cli.main()

        self.assertEqual(context.exception.code, 130)
        scorer.warmup.assert_called_once()
        scorer.close.assert_called_once()
        writer.finalize.assert_called_once()
        mock_validate_video.assert_called_once()


class CliParserTests(TestCase):
    def test_parser_accepts_instruction_override(self) -> None:
        args = cli.build_parser().parse_args(
            ["--repo-id", "demo/repo", "--output-dir", "/tmp/demo", "--instruction", "stack blocks"]
        )

        self.assertEqual(args.instruction, "stack blocks")

    def test_parser_defaults_to_gemini_backend(self) -> None:
        args = cli.build_parser().parse_args(
            ["--repo-id", "demo/repo", "--output-dir", "/tmp/demo"]
        )
        self.assertEqual(args.execution_backend, "gemini")

    def test_parser_accepts_gemini_model_override(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "--repo-id",
                "demo/repo",
                "--output-dir",
                "/tmp/demo",
                "--gemini-model",
                "gemini-3-flash-preview",
            ]
        )
        self.assertEqual(args.gemini_model, "gemini-3-flash-preview")


class CliInstructionOverrideTests(TestCase):
    @patch("lerobot_episode_scorer.cli.validate_video")
    @patch("lerobot_episode_scorer.cli.RollingOutputWriter")
    @patch("lerobot_episode_scorer.cli.GeminiVLMScorer")
    @patch("lerobot_episode_scorer.cli.EpisodeQualityScorer")
    @patch("lerobot_episode_scorer.cli.load_lerobot_dataset")
    @patch("lerobot_episode_scorer.cli.build_parser")
    def test_main_passes_instruction_override_to_execution_scorer(
        self,
        mock_build_parser: MagicMock,
        mock_load_dataset: MagicMock,
        mock_quality_scorer_class: MagicMock,
        mock_gemini_scorer_class: MagicMock,
        mock_writer_class: MagicMock,
        mock_validate_video: MagicMock,
    ) -> None:
        mock_build_parser.return_value.parse_args.return_value = SimpleNamespace(
            repo_id="demo/repo",
            root=None,
            output_dir=Path("/tmp/demo"),
            dataset_family="custom",
            camera_key=[],
            nominal_runtime_seconds=None,
            instruction="override instruction",
            execution_backend="gemini",
            gemini_model="gemini-flash-latest",
            lmstudio_model="demo-model",
            lmstudio_base_url="http://localhost:1234/v1",
            ollama_model="demo-ollama",
            ollama_host="http://localhost:11434",
            stitch_border_size=4,
            think=False,
            ollama_keep_alive=300.0,
            save_frames=False,
            execution_timeout_seconds=10.0,
        )
        episode = SimpleNamespace(
            episode_index=0,
            task="dataset instruction",
            label=None,
            cameras={"observation.images.top": MagicMock()},
        )
        mock_load_dataset.return_value = SimpleNamespace(
            repo_id="demo/repo",
            camera_keys=["observation.images.top"],
            nominal_runtime_seconds=1.0,
            episodes=[episode],
        )
        mock_quality_scorer_class.return_value.score_episode.return_value = {
            "aggregate": 0.5,
            "runtime_seconds": 1.0,
        }
        scorer = mock_gemini_scorer_class.return_value
        scorer.pre_extract_frames_for_episode.return_value = {}
        scorer.score_episode.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": None,
            "camera_used": "observation.images.top",
        }

        cli.main()

        scorer.score_episode.assert_called_once_with(
            episode,
            task="override instruction",
            pre_extracted={},
        )
        mock_writer_class.return_value.add_episode.assert_called_once()
        written_row = mock_writer_class.return_value.add_episode.call_args.args[0]
        self.assertEqual(written_row["task"], "override instruction")
        scorer.close.assert_called_once()
        mock_writer_class.return_value.finalize.assert_called_once()
        mock_validate_video.assert_called_once()

    @patch("lerobot_episode_scorer.cli.validate_video")
    @patch("lerobot_episode_scorer.cli.RollingOutputWriter")
    @patch("lerobot_episode_scorer.cli.LMStudioVLMScorer")
    @patch("lerobot_episode_scorer.cli.EpisodeQualityScorer")
    @patch("lerobot_episode_scorer.cli.load_lerobot_dataset")
    @patch("lerobot_episode_scorer.cli.build_parser")
    def test_main_still_allows_explicit_lmstudio_backend(
        self,
        mock_build_parser: MagicMock,
        mock_load_dataset: MagicMock,
        mock_quality_scorer_class: MagicMock,
        mock_lmstudio_scorer_class: MagicMock,
        mock_writer_class: MagicMock,
        mock_validate_video: MagicMock,
    ) -> None:
        mock_build_parser.return_value.parse_args.return_value = SimpleNamespace(
            repo_id="demo/repo",
            root=None,
            output_dir=Path("/tmp/demo"),
            dataset_family="custom",
            camera_key=[],
            nominal_runtime_seconds=None,
            instruction=None,
            execution_backend="lmstudio",
            gemini_model="gemini-flash-latest",
            lmstudio_model="demo-model",
            lmstudio_base_url="http://localhost:1234/v1",
            ollama_model="demo-ollama",
            ollama_host="http://localhost:11434",
            stitch_border_size=4,
            think=False,
            ollama_keep_alive=300.0,
            save_frames=False,
            execution_timeout_seconds=10.0,
        )
        episode = SimpleNamespace(
            episode_index=0,
            task="dataset instruction",
            label=None,
            cameras={"observation.images.top": MagicMock()},
        )
        mock_load_dataset.return_value = SimpleNamespace(
            repo_id="demo/repo",
            camera_keys=["observation.images.top"],
            nominal_runtime_seconds=1.0,
            episodes=[episode],
        )
        mock_quality_scorer_class.return_value.score_episode.return_value = {
            "aggregate": 0.5,
            "runtime_seconds": 1.0,
        }
        scorer = mock_lmstudio_scorer_class.return_value
        scorer.pre_extract_frames_for_episode.return_value = {}
        scorer.score_episode.return_value = {
            "score": 1.0,
            "probability": 1.0,
            "raw_response": "yes",
            "reasoning_trace": None,
            "camera_used": "observation.images.top",
        }

        cli.main()

        mock_lmstudio_scorer_class.assert_called_once_with(
            model="demo-model",
            base_url="http://localhost:1234/v1",
            border_size=4,
            think=False,
            timeout_seconds=10.0,
        )
        scorer.close.assert_called_once()
        mock_writer_class.return_value.finalize.assert_called_once()
        mock_validate_video.assert_called_once()
