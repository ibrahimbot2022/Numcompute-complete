import numpy as np
import matplotlib.pyplot as plt


def _finish_plot(save_path=None, show=True):
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return plt.gca()


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric", save_path=None, show=True):
    values = np.asarray(metric_values, dtype=float)
    plt.figure()
    plt.plot(np.arange(1, values.size + 1), values, marker="o")
    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    return _finish_plot(save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), title="Model comparison", ylabel="Metric", save_path=None, show=True):
    m1 = np.asarray(metric1, dtype=float)
    m2 = np.asarray(metric2, dtype=float)
    plt.figure()
    plt.plot(np.arange(1, m1.size + 1), m1, marker="o", label=labels[0])
    plt.plot(np.arange(1, m2.size + 1), m2, marker="x", label=labels[1])
    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    return _finish_plot(save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs Ground Truth", save_path=None, show=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    x = np.arange(y_true.size)
    plt.figure()
    plt.plot(x, y_true, marker="o", linestyle="", label="Ground truth")
    plt.plot(x, y_pred, marker="x", linestyle="", label="Prediction")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Class")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return _finish_plot(save_path, show)
