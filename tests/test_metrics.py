import unittest

from lerobot_episode_scorer.metrics import compute_binary_metrics


class MetricsTests(unittest.TestCase):
    def test_binary_metrics_on_separable_predictions(self) -> None:
        metrics = compute_binary_metrics(
            probabilities=[0.95, 0.8, 0.3, 0.1],
            labels=[1, 1, 0, 0],
        )

        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["auprc"], 1.0)

    def test_binary_metrics_allow_single_class_auc_to_be_missing(self) -> None:
        metrics = compute_binary_metrics(probabilities=[0.9, 0.7], labels=[1, 1])
        self.assertIsNone(metrics["auroc"])
        self.assertAlmostEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
