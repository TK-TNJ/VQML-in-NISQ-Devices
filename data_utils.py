"""
Shared Data Utilities for VQML Benchmark
=========================================
Common configuration, data loading (with PCA feature reduction),
metrics display, and prediction visualization shared across all algorithm files.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import pennylane as qml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================
TRAIN_SIZE = 1000
TEST_SIZE  = 250
N_QUBITS   = 4       # PCA reduces to 4 principal components
EPOCHS     = 20
SAVED_DIR  = os.path.join(os.path.dirname(__file__), "saved_models")

# Ensure saved_models directory exists
os.makedirs(SAVED_DIR, exist_ok=True)

# Quantum device
dev = qml.device("default.qubit", wires=N_QUBITS)

# Label mapping: class 0 = digit 2, class 1 = digit 7
LABEL_MAP = {0: "Digit 2", 1: "Digit 7"}


# ============================================================
# DATA LOADING (PCA-based feature reduction)
# ============================================================
def load_data():
    """
    Download MNIST, filter digits 2 vs 7, apply PCA to reduce
    784 dimensions -> 4 principal components, normalize to [0, pi].
    """
    print("\n[*] Loading & Processing MNIST Data...")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Filter for digits 2 and 7 only
    idx = (dataset.targets == 2) | (dataset.targets == 7)
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]

    # Relabel: 2 -> 0, 7 -> 1
    dataset.targets = torch.where(dataset.targets == 2, 0, 1)

    # Keep original 28x28 images for visualization later
    original_images = dataset.data.clone()

    # Flatten 28x28 -> 784
    flat_data = dataset.data.float().view(-1, 784).numpy()

    # Apply PCA: 784 features -> 4 principal components
    pca = PCA(n_components=N_QUBITS)
    pca_data = pca.fit_transform(flat_data)
    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"   [PCA] Reduced 784 dims -> {N_QUBITS} components ({explained:.1f}% variance retained)")

    # Normalize PCA output to [0, pi] for angle encoding
    pca_min = pca_data.min(axis=0)
    pca_max = pca_data.max(axis=0)
    pca_normalized = (pca_data - pca_min) / (pca_max - pca_min + 1e-8) * np.pi
    pca_tensor = torch.tensor(pca_normalized, dtype=torch.float32)

    # Split
    X_train = pca_tensor[:TRAIN_SIZE]
    y_train = dataset.targets[:TRAIN_SIZE]
    X_test  = pca_tensor[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]
    y_test  = dataset.targets[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]

    # Original images for visualization (test set)
    orig_test = original_images[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]

    print(f"   [OK] {len(X_train)} train / {len(X_test)} test samples (Digits 2 vs 7)")
    return X_train, y_train, X_test, y_test, orig_test


# ============================================================
# METRICS DISPLAY
# ============================================================
def compute_metrics(y_true, y_pred):
    """Compute and return accuracy, precision, recall, F1."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1


def show_metrics(algo_name, y_true, y_pred, elapsed=None):
    """Print a formatted metrics table."""
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  [RESULTS] EVALUATION RESULTS -- {algo_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if elapsed:
        print(f"  Time      : {elapsed:.1f}s")
    print(f"{'='*50}")
    return acc, prec, rec, f1


# ============================================================
# PREDICTION VISUALIZATION
# ============================================================
def show_predictions(algo_name, orig_images, y_true, y_pred, n=10):
    """
    Display a grid of test images with predicted vs true labels.
    Shows first n images with [OK] (correct) or [X] (wrong) markers.
    """
    n = min(n, len(y_true))
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle(f"{algo_name} -- Predictions on Test Set", fontsize=14, fontweight='bold')

    if rows == 1:
        axes = [axes]

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[0][i % cols] if cols > 1 else axes[i]
        if i < n:
            img = orig_images[i].numpy()
            true_label = int(y_true[i])
            pred_label = int(y_pred[i])
            correct = true_label == pred_label

            ax.imshow(img, cmap='gray')
            marker = "[OK]" if correct else "[X]"
            true_digit = LABEL_MAP[true_label]
            pred_digit = LABEL_MAP[pred_label]
            ax.set_title(f"{marker} Pred: {pred_digit}\nTrue: {true_digit}",
                         fontsize=9, color='green' if correct else 'red')
        ax.axis('off')

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(SAVED_DIR, f"{algo_name.lower()}_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[IMG] Prediction image saved to: {save_path}")
    try:
        plt.show()
    except Exception:
        pass
