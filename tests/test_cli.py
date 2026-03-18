from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from lerobot_episode_scorer import cli


class CliInterruptTests(TestCase):
    @patch("lerobot_episode_scorer.cli.validate_video")
    @patch("lerobot_episode_scorer.cli.RollingOutputWriter")
    @patch("lerobot_episode_scorer.cli.LMStudioVLMScorer")
    @patch("lerobot_episode_scorer.cli.EpisodeQualityScorer")
    @patch("lerobot_episode_scorer.cli.load_lerobot_dataset")
    @patch("lerobot_episode_scorer.cli.build_parser")
    def test_keyboard_interrupt_is_reraised_after_flushing_outputs(
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
            execution_backend="lmstudio",
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
        scorer = mock_lmstudio_scorer_class.return_value
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
