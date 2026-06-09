import unittest
from xml.parsers.expat import model
import numpy as np

from numcompute.tree import DecisionTreeClassifier
from numcompute.stream import StreamTrainer


class TestStreamTrainer(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [3.0, 3.0],
        ])

        self.y = np.array([
            0, 0, 0, 1, 1, 1, 1, 1
        ])

    def test_fit_chunk_updates_model(self):
        model = DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        )

        trainer = StreamTrainer(model)

        trainer.fit_chunk(self.X[:4], self.y[:4])

        preds = model.predict(self.X[:4])

        self.assertEqual(preds.shape, self.y[:4].shape)

    def test_score_chunk_returns_float(self):
        model = DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        )

        trainer = StreamTrainer(model)

        trainer.fit_chunk(self.X[:4], self.y[:4])

        score = trainer.score_chunk(
            self.X[:4],
            self.y[:4]
        )

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_partial_stream_two_chunks(self):
        model = DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        )

        trainer = StreamTrainer(model)

        trainer.fit_chunk(self.X[:4], self.y[:4])
        trainer.fit_chunk(self.X[4:], self.y[4:])

        preds = model.predict(self.X)

        self.assertEqual(preds.shape, self.y.shape)

    def test_logs_are_updated_after_fit(self):
        model = DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        )

        trainer = StreamTrainer(model)

        trainer.fit_chunk(self.X[:4], self.y[:4])
        trainer.score_chunk(self.X[:4], self.y[:4])

        self.assertTrue(hasattr(trainer, "logs"))
        self.assertGreaterEqual(len(trainer.logs), 1)

    def test_predict_before_fit_raises_through_model(self):
        model = DecisionTreeClassifier()
        trainer = StreamTrainer(model)

        with self.assertRaises(RuntimeError):
            trainer.score_chunk(self.X, self.y)

    def test_shape_mismatch_raises(self):
        model = DecisionTreeClassifier()
        trainer = StreamTrainer(model)

        with self.assertRaises(ValueError):
            trainer.fit_chunk(self.X, self.y[:3])
            
            
    def test_empty_chunk_raises(self):
        model = DecisionTreeClassifier()

        trainer = StreamTrainer(model)

        with self.assertRaises(ValueError):
            trainer.fit_chunk(
            np.empty((0, 2)),
            np.array([])
        )
    def test_score_returns_float_after_multiple_updates(self):
        model = DecisionTreeClassifier()

        trainer = StreamTrainer(model)

        trainer.fit_chunk(self.X[:4], self.y[:4])
        trainer.fit_chunk(self.X[4:], self.y[4:])

        score = trainer.score_chunk(
        self.X,
        self.y
    )

        self.assertIsInstance(score, float)

if __name__ == "__main__":
    unittest.main()