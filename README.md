# NumCompute

NumCompute is a modular machine learning framework built entirely from **Python and NumPy**. It replicates core functionality found in machine learning libraries such as scikit-learn while emphasizing numerical computation, vectorisation, software engineering practices, and algorithm implementation from first principles.

---

## Features

### Core Framework

- CSV data loading with missing value handling
- Data preprocessing (StandardScaler, MinMaxScaler, OneHotEncoder)
- Sorting and searching algorithms (top-k, quickselect, binary search)
- Ranking with tie handling and percentile computation
- Statistical functions (mean, median, standard deviation, histogram, quantiles)
- Evaluation metrics (accuracy, precision, recall, F1-score, MSE, confusion matrix)
- Numerical optimisation (finite-difference gradient and Jacobian)
- Pipeline abstraction for chaining transformations
- Utility functions (activations, distances, batching, logsumexp)
- Benchmarking (loop vs vectorised performance comparison)

### Assignment 2.2 Extensions

- Decision Tree classifier implementation
- Ensemble classifier using multiple decision trees
- Streaming / incremental learning support
- Stream-based model training and evaluation
- Performance visualisation utilities
- Prequential evaluation for streaming workflows

---

## Project Structure

```text
NumCompute/
├── numcompute/
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stat.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   ├── benchmarking.py
│   ├── tree.py
│   ├── ensemble.py
│   ├── stream.py
│   └── visualise.py
│
├── tests/
│   ├── test_io.py
│   ├── test_preprocessing.py
│   ├── test_sort_search.py
│   ├── test_rank.py
│   ├── test_stat.py
│   ├── test_metrics.py
│   ├── test_optim.py
│   ├── test_pipeline.py
│   ├── test_utils.py
│   ├── test_benchmarking.py
│   ├── test_tree.py
│   ├── test_ensemble.py
│   ├── test_stream.py
│   └── test_visualise.py
│
├── demo/
│   ├── quickstart.ipynb
│   └── stream_demo.ipynb
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ibrahimbot2022/Numcompute-complete.git
cd Numcompute-complete
```

Install the package locally:

```bash
pip install -e .
```

---

## Running the Demo

### Quickstart Demonstration

```bash
cd demo
jupyter notebook quickstart.ipynb
```

### Streaming Learning Demonstration

```bash
cd demo
jupyter notebook stream_demo.ipynb
```

---

## Running Tests

Run all tests using:

```bash
python -m unittest discover -s tests
```

The test suite covers:

- Core algorithms
- Edge cases (NaN values, ties, duplicates)
- Integration between modules
- Decision tree classification
- Ensemble learning
- Streaming workflows
- Visualisation utilities

Current status:

```text
Ran 187 tests

OK
```

---

## Benchmark Results

| Metric | Loop Time (s) | Vectorised Time (s) | Speedup |
|----------|-------------|-------------------|---------|
| MSE | ~0.03 | ~0.001 | ~30x |
| Accuracy | ~0.02 | ~0.0004 | ~50x |

Vectorised implementations provide significant performance improvements compared to traditional loop-based implementations.

---

## Design Highlights

- Fully vectorised computations using NumPy
- Numerical stability through logsumexp and stable softmax implementations
- Modular architecture with a consistent API design
- Clear input/output shape validation
- Extensible framework design
- Streaming machine learning support

---

## Notes

- No external machine learning or deep learning libraries were used
- All algorithms were implemented from first principles
- Designed to demonstrate numerical computing and machine learning fundamentals
- Structured to resemble real-world machine learning libraries

---

## Usage Note

The notebooks are designed to run within the project structure. If running from another location, install the package first:

```bash
pip install -e .
```

---

## Authors

Original NumCompute framework developed as part of a group project.

This repository contains the complete framework together with the Assignment 2.2 extensions including decision trees, ensemble learning, streaming workflows, visualisation utilities, additional tests, and demonstration notebooks.