import unittest
import os
import tempfile
from unittest import result
import numpy as np

import matplotlib
matplotlib.use("Agg")

from numcompute.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
)


def get_fig(result):
    if isinstance(result, tuple):
        result = result[0]

    if hasattr(result, "savefig"):
        return result

    if hasattr(result, "figure"):
        return result.figure

    return result


class TestVisualise(unittest.TestCase):

    def test_plot_metric_over_time_returns_figure(self):
        result = plot_metric_over_time(
            [0.5, 0.6, 0.7, 0.8],
            title="Accuracy Over Time",
            ylabel="Accuracy",
            show=False
        )

        fig = get_fig(result)

        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, "savefig"))

    def test_compare_models_returns_figure(self):
        result = compare_models(
            [0.5, 0.6, 0.7],
            [0.4, 0.55, 0.75],
            labels=("Tree", "Ensemble"),
            show=False
        )

        fig = get_fig(result)

        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, "savefig"))

    def test_predictions_vs_ground_truth_returns_figure(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        result = plot_predictions_vs_ground_truth(
            y_true,
            y_pred,
            show=False
        )

        fig = get_fig(result)

        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, "savefig"))

    def test_plot_metric_over_time_saves_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metric.png")

            result = plot_metric_over_time(
                [0.5, 0.6, 0.7],
                title="Accuracy",
                ylabel="Accuracy",
                save_path=path,
                show=False
            )

            fig = get_fig(result)

            self.assertIsNotNone(fig)
            self.assertTrue(os.path.exists(path))

    def test_compare_models_saves_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "compare.png")

            result = compare_models(
                [0.5, 0.6, 0.7],
                [0.4, 0.55, 0.75],
                labels=("Tree", "Ensemble"),
                save_path=path,
                show=False
            )

            fig = get_fig(result)

            self.assertIsNotNone(fig)
            self.assertTrue(os.path.exists(path))

    def test_predictions_vs_ground_truth_saves_file(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "predictions.png")

            result = plot_predictions_vs_ground_truth(
                y_true,
                y_pred,
                save_path=path,
                show=False
            )

            fig = get_fig(result)

            self.assertIsNotNone(fig)
            self.assertTrue(os.path.exists(path))

    def test_predictions_shape_mismatch_raises(self):
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1])

        with self.assertRaises(ValueError):
            plot_predictions_vs_ground_truth(
                y_true,
                y_pred,
                show=False
            )
            
    def test_empty_metric_list_returns_figure(self):
        result = plot_metric_over_time(
             [],
            title="Accuracy",
            ylabel="Accuracy",
            show=False
        )

        fig = get_fig(result)

        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, "savefig"))  


if __name__ == "__main__":
    unittest.main()