import numpy as np


class DecisionTreeClassifier:
    """Depth-limited NumPy decision tree classifier for batch and streaming use.

    partial_fit stores observed chunks and rebuilds the tree. This keeps the API
    stream-compatible while preserving deterministic, easy-to-test behaviour.
    """

    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini", max_features=None, random_state=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.fitted = False
        self._X_seen = None
        self._y_seen = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X, y = self._validate_xy(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = self._feature_medians(X)
        X = self._replace_nan(X)
        self.root_ = self._build_tree(X, y, depth=0)
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
            raise RuntimeError("DecisionTreeClassifier: call fit() or partial_fit() before predict().")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be 1D or 2D.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features but got {X.shape[1]}.")
        X = self._replace_nan(X)
        return np.asarray([self._predict_row(row, self.root_) for row in X])

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
            raise ValueError("DecisionTreeClassifier cannot fit empty data.")
        return X, y

    def _feature_medians(self, X):
        med = np.nanmedian(X, axis=0)
        return np.where(np.isnan(med), 0.0, med)

    def _replace_nan(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X[inds] = self.feature_medians_[inds[1]]
        return X

    def _impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        if self.criterion == "gini":
            return 1.0 - np.sum(p * p)
        return -np.sum(p * np.log2(p + 1e-12))

    def _majority_class(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _feature_indices(self, n_features):
        mf = self.max_features
        if mf is None:
            return np.arange(n_features)
        if mf == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif mf == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(mf, float):
            if not (0 < mf <= 1):
                raise ValueError("float max_features must be in (0, 1].")
            k = max(1, int(np.ceil(mf * n_features)))
        else:
            k = int(mf)
        k = min(max(1, k), n_features)
        return self.rng.choice(n_features, size=k, replace=False)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in self._feature_indices(n_features):
            values = np.unique(X[:, feature])
            if values.size <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left == 0 or n_right == 0:
                    continue
                child_impurity = (n_left / n_samples) * self._impurity(y[left_mask]) + (n_right / n_samples) * self._impurity(y[right_mask])
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        node = {"prediction": self._majority_class(y), "n_samples": int(y.size)}
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or y.size < self.min_samples_split
            or np.unique(y).size == 1
        ):
            node["leaf"] = True
            return node

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 0:
            node["leaf"] = True
            return node

        left_mask = X[:, feature] <= threshold
        node.update({
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "gain": gain,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        })
        return node

    def _predict_row(self, row, node):
        while not node.get("leaf", False):
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"]
