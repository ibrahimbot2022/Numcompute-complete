import numpy as np
from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    """Bagging/Random-Forest style ensemble of decision trees."""

    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, criterion="gini", max_features="sqrt", bootstrap=True, random_state=None):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.fitted = False
        self._X_seen = None
        self._y_seen = None

    def fit(self, X, y):
        X, y = self._validate_xy(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.trees_ = []
        n = X.shape[0]
        for i in range(self.n_estimators):
            if self.bootstrap:
                idx = self.rng.integers(0, n, size=n)
            else:
                idx = np.arange(n)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=None if self.random_state is None else self.random_state + i + 1,
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        self.fitted = True
        return self

    def partial_fit(self, X, y, classes=None):
        X, y = self._validate_xy(X, y)
        if self._X_seen is None:
            self._X_seen = X.copy()
            self._y_seen = y.copy()
        else:
            if X.shape[1] != self._X_seen.shape[1]:
                raise ValueError(f"expected {self._X_seen.shape[1]} features but got {X.shape[1]}.")
            self._X_seen = np.vstack((self._X_seen, X))
            self._y_seen = np.concatenate((self._y_seen, y))
        if classes is not None:
            self.classes_ = np.asarray(classes)
        return self.fit(self._X_seen, self._y_seen)

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("EnsembleClassifier: call fit() or partial_fit() before predict().")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be 1D or 2D.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features but got {X.shape[1]}.")
        votes = np.vstack([tree.predict(X) for tree in self.trees_]).T
        preds = []
        for row in votes:
            labels, counts = np.unique(row, return_counts=True)
            preds.append(labels[np.argmax(counts)])
        return np.asarray(preds)

    def _validate_xy(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 1D or 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("EnsembleClassifier cannot fit empty data.")
        return X, y


RandomForestClassifier = EnsembleClassifier
