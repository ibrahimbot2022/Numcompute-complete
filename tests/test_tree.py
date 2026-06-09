import unittest
from xml.parsers.expat import model
import numpy as np

from numcompute.tree import DecisionTreeClassifier


class TestDecisionTreeClassifier(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
        ])

        self.y = np.array([
            0, 0, 1, 1, 1, 1
        ])

    def test_fit_predict_basic(self):
        model = DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        )

        model.fit(self.X, self.y)

        preds = model.predict(self.X)

        self.assertEqual(preds.shape, self.y.shape)
        self.assertGreaterEqual(
            np.mean(preds == self.y),
            0.8
        )

    def test_partial_fit_two_chunks(self):
        model = DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        )

        model.partial_fit(
            self.X[:3],
            self.y[:3]
        )

        model.partial_fit(
            self.X[3:],
            self.y[3:]
        )

        preds = model.predict(self.X)

        self.assertEqual(preds.shape, self.y.shape)
        self.assertGreaterEqual(
            np.mean(preds == self.y),
            0.8
        )

    def test_predict_before_fit_raises(self):
        model = DecisionTreeClassifier()

        with self.assertRaises(RuntimeError):
            model.predict(self.X)

    def test_invalid_criterion_raises(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(
                criterion="wrong"
            )

    def test_shape_mismatch_raises(self):
        model = DecisionTreeClassifier()

        with self.assertRaises(ValueError):
            model.fit(
                self.X,
                self.y[:3]
            )

    def test_nan_values_are_handled(self):
        X = self.X.copy()
        X[0, 0] = np.nan

        model = DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        )

        model.fit(X, self.y)

        preds = model.predict(X)

        self.assertEqual(preds.shape, self.y.shape)

    def test_single_feature_input(self):
        X = np.array([
            [0],
            [1],
            [2],
            [3],
            [4],
            [5]
        ])

        y = np.array([
            0,
            0,
            0,
            1,
            1,
            1
        ])

        model = DecisionTreeClassifier(
            max_depth=2
        )

        model.fit(X, y)

        preds = model.predict(X)

        self.assertEqual(preds.shape, y.shape)
        self.assertGreaterEqual(
            np.mean(preds == y),
            0.8
        )

    def test_wrong_feature_count_raises(self):
        model = DecisionTreeClassifier(
            max_depth=2
        )

        model.fit(self.X, self.y)

        bad_X = np.array([
            [1.0, 2.0, 3.0]
        ])

        with self.assertRaises(ValueError):
            model.predict(bad_X)
            
    def test_same_predictions_multiple_calls(self):
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(self.X, self.y)

        p1 = model.predict(self.X)
        p2 = model.predict(self.X)

        np.testing.assert_array_equal(p1, p2)

    def test_single_class_training(self):
        y = np.zeros(len(self.X), dtype=int)

        model = DecisionTreeClassifier(max_depth=3)
        model.fit(self.X, y)

        preds = model.predict(self.X)

        np.testing.assert_array_equal(preds, y)
        
        
if __name__ == "__main__":
    unittest.main()