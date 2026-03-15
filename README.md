# ⟨ψ| VQML Benchmark — Variational Quantum Machine Learning on NISQ Devices

A comprehensive benchmark of **5 Variational Quantum Machine Learning (VQML) algorithms** for binary image classification on the MNIST dataset. This project evaluates quantum circuit-based classifiers running on simulated **NISQ (Noisy Intermediate-Scale Quantum)** hardware using PennyLane and PyTorch.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Results](#performance-results)
- [Frontend Dashboard](#frontend-dashboard)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project benchmarks five distinct variational quantum machine learning approaches on a binary classification task:

- **Task**: Classify handwritten digits **2 vs 7** from the MNIST dataset
- **Approach**: Hybrid quantum-classical — quantum circuits process data, classical optimizers tune parameters
- **Qubits**: 4 qubits (matching 4 PCA-reduced features)
- **Framework**: PennyLane for quantum circuits, PyTorch for classical optimization
- **Evaluation**: Accuracy, Precision, Recall, and F1-Score

### Why This Project?

Quantum computing promises computational advantages for certain tasks, but today's quantum hardware (NISQ devices) is noisy and limited. **Variational quantum algorithms** are designed to work within these constraints by combining parameterized quantum circuits with classical optimization. This benchmark explores how different quantum circuit architectures perform on a well-understood classification problem.

---

## Project Structure

```
VQML/
├── frontend/                  # Interactive web dashboard
│   ├── index.html             # Main HTML (About, Pipeline, Algorithms, Results, Glossary)
│   ├── styles.css             # Professional pastel-light theme
│   └── script.js              # Charts, animations, interactive demo
│
├── data/                      # MNIST dataset (auto-downloaded)
│   └── MNIST/                 # Raw MNIST data files
│
├── saved_models/              # Trained weights & prediction images
│   ├── qnn_weights.pt         # QNN trained parameters
│   ├── qcnn_weights.pt        # QCNN trained parameters
│   ├── vqc_weights.pt         # VQC trained parameters
│   ├── vqfe_weights.pt        # VQFE trained parameters
│   ├── qsvm_model.joblib      # QSVM trained SVC model
│   ├── qsvm_train_data.pt     # QSVM training subset
│   ├── qnn_predictions.png    # QNN prediction visualization
│   ├── qcnn_predictions.png   # QCNN prediction visualization
│   ├── vqc_predictions.png    # VQC prediction visualization
│   ├── vqfe_predictions.png   # VQFE prediction visualization
│   └── qsvm_predictions.png   # QSVM prediction visualization
│
├── data_utils.py              # Shared utilities (data loading, PCA, metrics)
├── qnn.py                     # Quantum Neural Network implementation
├── qcnn.py                    # Quantum Convolutional Neural Network implementation
├── vqc.py                     # Variational Quantum Classifier implementation
├── vqfe.py                    # Variational Quantum Feature Embedding implementation
├── qsvm.py                    # Quantum Support Vector Machine implementation
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## Algorithms

### 1. QNN — Quantum Neural Network
The simplest variational approach using `AngleEmbedding` + 3 layers of `BasicEntanglerLayers`.
- **Encoding**: AngleEmbedding (RX rotations)
- **Variational**: BasicEntanglerLayers (single-qubit rotations + nearest-neighbor CNOTs)
- **Parameters**: 12 (3 layers × 4 qubits)

### 2. QCNN — Quantum Convolutional Neural Network
Mimics classical CNNs with convolutional and pooling layers.
- **Conv Layer**: Pairs of RX/RY gates + CNOT entangling
- **Pooling Layer**: Rot gate on qubit 0
- **Parameters**: 6

### 3. VQC — Variational Quantum Classifier
The most expressive ansatz using `StronglyEntanglingLayers`.
- **Encoding**: AngleEmbedding
- **Variational**: StronglyEntanglingLayers (3 rotations per qubit + long-range CNOTs)
- **Parameters**: 36 (3 layers × 4 qubits × 3 rotations)

### 4. VQFE — Variational Quantum Feature Embedding
Unique approach with trainable feature embedding.
- **Encoding**: Trainable RY rotations (input × learnable weight per qubit)
- **Variational**: BasicEntanglerLayers (3 layers)
- **Parameters**: 16 (4 embedding weights + 12 variational)

### 5. QSVM — Quantum Support Vector Machine
Kernel-based approach using quantum circuit fidelity.
- **Method**: Quantum fidelity kernel |⟨φ(x₁)|φ(x₂)⟩|²
- **Classifier**: Classical SVC with precomputed kernel
- **Complexity**: O(n²) kernel computation

---

## Prerequisites

- **Python** 3.8 or higher
- **pip** (Python package manager)
- ~500 MB disk space (for MNIST dataset and dependencies)
- Internet connection (for first-time MNIST download)

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd VQML
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pennylane torch torchvision scikit-learn numpy matplotlib joblib
```

Or install all at once:

```bash
pip install pennylane torch torchvision scikit-learn numpy matplotlib joblib
```

### 4. Verify Installation

```bash
python -c "import pennylane as qml; import torch; print('PennyLane:', qml.__version__); print('PyTorch:', torch.__version__)"
```

---

## Usage

Each algorithm file (`qnn.py`, `qcnn.py`, `vqc.py`, `vqfe.py`, `qsvm.py`) supports three modes:

### Training

Train the model and save weights to `saved_models/`:

```bash
python qnn.py --train
python qcnn.py --train
python vqc.py --train
python vqfe.py --train
python qsvm.py --train
```

### Evaluation

Load saved weights and display classification metrics:

```bash
python qnn.py --evaluate
python qcnn.py --evaluate
python vqc.py --evaluate
python vqfe.py --evaluate
python qsvm.py --evaluate
```

### Prediction Visualization

Generate and save prediction images showing test set results:

```bash
python qnn.py --predict
python qcnn.py --predict
python vqc.py --predict
python vqfe.py --predict
python qsvm.py --predict
```

### Run All Modes at Once

```bash
python vqc.py --train --evaluate --predict
```

### Run All Algorithms

```bash
# Train all (this will take several minutes)
for algo in qnn qcnn vqc vqfe qsvm; do python ${algo}.py --train; done

# Evaluate all
for algo in qnn qcnn vqc vqfe qsvm; do python ${algo}.py --evaluate; done
```

**Windows PowerShell:**
```powershell
# Train all
foreach ($algo in @("qnn", "qcnn", "vqc", "vqfe", "qsvm")) { python "$algo.py" --train }

# Evaluate all
foreach ($algo in @("qnn", "qcnn", "vqc", "vqfe", "qsvm")) { python "$algo.py" --evaluate }
```

---

## Performance Results

| Algorithm | Accuracy | Precision | Recall | F1-Score | Notes |
|-----------|----------|-----------|--------|----------|-------|
| **VQC**   | 96.4%    | 94.2%     | 99.2%  | **96.6%**| **Best Overall Performer** |
| **QCNN**  | 96.4%    | 94.2%     | 99.2%  | **96.6%**| Tied for best, fewer parameters |
| **QSVM**  | 90.0%    | 92.8%     | 92.8%  | 92.8%    | Strong baseline, O(n²) kernel limit |
| **VQFE**  | 57.6%    | 59.1%     | 63.6%  | 61.3%    | Adaptive embedding struggles here |
| **QNN**   | 52.8%    | 52.8%     | 100.0% | 69.1%    | Predicts all as class 1 |

### Key Findings

1. **VQC and QCNN achieve the best overall performance** (F1: 96.6%, Acc: 96.4%). Both highly expressive architectures succeed at capturing the features.
2. **QCNN is highly efficient**, matching VQC's performance while using dramatically fewer parameters due to its hierarchical pooling structure.
3. **QNN fails to learn** (52.8% accuracy) — BasicEntanglerLayers alone lack sufficient expressivity.
4. **QSVM provides a strong baseline** — excellent kernel-based performance despite a smaller computational subset.
5. **Circuit design matters more than parameter count** — architecture choice is the primary driver of performance

---

## Frontend Dashboard

The project includes an interactive web dashboard for exploring results.

### Launching the Dashboard

Simply open `frontend/index.html` in any modern web browser:

```bash
# Option 1: Direct file open
start frontend/index.html        # Windows
open frontend/index.html          # macOS
xdg-open frontend/index.html     # Linux

# Option 2: Python HTTP server (recommended)
cd frontend
python -m http.server 8000
# Then visit http://localhost:8000
```

### Dashboard Features

- **About This Research** — Project overview for newcomers
- **How It Works** — Step-by-step pipeline visualization
- **Algorithm Explorer** — Interactive circuit diagrams with code snippets
- **Performance Breakdown** — Animated circular progress rings per algorithm
- **Performance Dashboard** — Bar chart, radar chart, and data table views
- **Key Findings** — Comparative insights and analysis
- **Prediction Gallery** — Test set prediction visualizations for each algorithm
- **Interactive Demo** — Draw a digit and see simulated classification
- **Quantum Glossary** — Expandable definitions of quantum computing terms
- **Technical Details** — Configuration, pipeline, training, and dependency info

---

## Configuration

All shared configuration is in `data_utils.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TRAIN_SIZE` | 1,000 | Number of training samples |
| `TEST_SIZE` | 250 | Number of test samples |
| `N_QUBITS` | 4 | Number of qubits (= PCA components) |
| `EPOCHS` | 20 | Training epochs for variational algorithms |
| `SAVED_DIR` | `saved_models/` | Directory for saved weights/models |

QSVM-specific settings in `qsvm.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TRAIN_SUB` | 40 | Training subset for kernel computation |
| `TEST_SUB` | 20 | Test subset for kernel computation |

---

## How It Works

### Data Pipeline

1. **Load MNIST** — Download 28×28 grayscale handwritten digit images
2. **Filter** — Keep only digits 2 and 7; relabel to 0 and 1
3. **Flatten** — Reshape 28×28 images → 784-dimensional vectors
4. **PCA** — Reduce 784 dimensions → 4 principal components (~50% variance retained)
5. **Normalize** — Scale to [0, π] for quantum angle encoding
6. **Split** — 1,000 training / 250 test samples

### Quantum Circuit Flow

1. **Angle Encoding** — Each of the 4 PCA components becomes a rotation angle on one qubit
2. **Variational Layers** — Parameterized quantum gates process the encoded data
3. **Measurement** — PauliZ expectation on qubit 0 gives output ∈ [-1, +1]
4. **Decision** — Output > 0 → class 1 (digit 7), else class 0 (digit 2)

### Training Loop

- **Loss**: Mean Squared Error between circuit output and target (±1)
- **Optimizer**: PyTorch Adam (lr=0.1) with automatic differentiation through quantum circuits
- **Target Encoding**: Labels mapped as 0 → -1, 1 → +1

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pennylane` | Quantum circuit framework and simulation |
| `torch` | Neural network framework and autograd |
| `torchvision` | MNIST dataset loading |
| `scikit-learn` | PCA, SVC classifier, evaluation metrics |
| `numpy` | Numerical operations |
| `matplotlib` | Prediction visualization plots |
| `joblib` | QSVM model serialization |

---

## Troubleshooting

### Common Issues

**MNIST download fails:**
```bash
# Ensure internet connection, or manually download to data/MNIST/
```

**Out of memory during training:**
```bash
# Reduce TRAIN_SIZE in data_utils.py (e.g., 500)
```

**QSVM training is very slow:**
```bash
# This is expected due to O(n²) kernel computation
# Reduce TRAIN_SUB in qsvm.py (e.g., 20)
```

**`No saved weights found!` error:**
```bash
# Run --train before --evaluate or --predict
python vqc.py --train
```

**Frontend images not loading:**
```bash
# Ensure saved_models/ contains prediction PNG files
# Run --predict for each algorithm first
```

---

## License

This project is part of an academic research paper implementation on Variational Quantum Machine Learning algorithms for NISQ devices.

---

<div align="center">

**Built with** PennyLane • PyTorch • scikit-learn • Python

⟨ψ| VQML Benchmark

</div>
