import sys
import numpy as np
from .metrics import accuracy


class StreamTrainer:
    """Manage chunk-wise training, scoring, and metric logging."""

    def __init__(self, model, pipeline=None, metric_fn=accuracy):
        self.model = model
        self.pipeline = pipeline
        self.metric_fn = metric_fn
        self.logs = {"chunk": [], "chunk_accuracy": [], "cumulative_accuracy": [], "memory_bytes": []}
        self._seen_true = []
        self._seen_pred = []

    def fit_chunk(self, X, y):
        if self.pipeline is not None:
            self.pipeline.partial_fit(X, y)
        else:
            if not callable(getattr(self.model, "partial_fit", None)):
                raise TypeError("model must implement partial_fit for streaming training.")
            self.model.partial_fit(X, y)
        return self

    def score_chunk(self, X, y):
        if self.pipeline is not None:
            y_pred = self.pipeline.predict(X)
        else:
            y_pred = self.model.predict(X)
        chunk_score = float(self.metric_fn(y, y_pred))
        self._seen_true.extend(np.asarray(y).ravel().tolist())
        self._seen_pred.extend(np.asarray(y_pred).ravel().tolist())
        cumulative = float(self.metric_fn(np.asarray(self._seen_true), np.asarray(self._seen_pred)))
        self.logs["chunk"].append(len(self.logs["chunk"]) + 1)
        self.logs["chunk_accuracy"].append(chunk_score)
        self.logs["cumulative_accuracy"].append(cumulative)
        self.logs["memory_bytes"].append(self.memory_footprint())
        return chunk_score

    def fit_score_chunk(self, X, y):
        self.fit_chunk(X, y)
        return self.score_chunk(X, y)

    def memory_footprint(self):
        total = sys.getsizeof(self.model)
        if self.pipeline is not None:
            total += sys.getsizeof(self.pipeline)
        return int(total)

    def history(self):
        return {k: np.asarray(v) for k, v in self.logs.items()}
