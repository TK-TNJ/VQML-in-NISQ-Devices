"""
QSVM - Quantum Support Vector Machine
========================================
Binary MNIST classification (Digit 2 vs 7) using PCA feature reduction
and a quantum kernel (fidelity-based) with classical SVM.

Usage:
    python qsvm.py --train       # Train & save model
    python qsvm.py --evaluate    # Load model & show metrics
    python qsvm.py --predict     # Load model & show predicted images
"""

import argparse
import torch
import pennylane as qml
import numpy as np
import time
import os
import joblib

from sklearn.svm import SVC
from data_utils import (
    load_data, dev, N_QUBITS, SAVED_DIR,
    show_metrics, show_predictions
)

ALGO_NAME     = "QSVM"
MODEL_FILE    = os.path.join(SAVED_DIR, "qsvm_model.joblib")
TRAIN_DATA_FILE = os.path.join(SAVED_DIR, "qsvm_train_data.pt")

# Use smaller subsets for kernel computation (O(n^2) complexity)
TRAIN_SUB = 40
TEST_SUB  = 20


# ============================================================
# KERNEL CIRCUIT
# ============================================================
@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(N_QUBITS))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))


def qsvm_kernel(x1, x2):
    """Fidelity-based quantum kernel: |<phi(x1)|phi(x2)>|^2"""
    return kernel_circuit(x1, x2)[0]


def compute_kernel_matrix(A, B):
    """Compute NxM kernel matrix between datasets A and B."""
    n = len(A)
    m = len(B)
    K = np.zeros((n, m))
    total = n * m
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            K[i, j] = qsvm_kernel(a, b)
        if (i + 1) % 10 == 0:
            print(f"   Computing kernel: {(i+1)*m}/{total} entries done...")
    return K


# ============================================================
# SECTION 1: TRAINING
# ============================================================
def train():
    print(f"\n{'='*50}")
    print(f"  [TRAIN] TRAINING -- {ALGO_NAME}")
    print(f"{'='*50}")

    X_train, y_train, X_test, y_test, orig_test = load_data()

    X_tr_sub = X_train[:TRAIN_SUB]
    y_tr_sub = y_train[:TRAIN_SUB]

    start_time = time.time()

    print(f"\n   Computing {TRAIN_SUB}x{TRAIN_SUB} training kernel matrix...")
    K_train = compute_kernel_matrix(X_tr_sub, X_tr_sub)

    print(f"   Training SVM with precomputed quantum kernel...")
    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_tr_sub.numpy())

    elapsed = time.time() - start_time
    print(f"\n   [OK] Training complete in {elapsed:.1f}s")

    joblib.dump(clf, MODEL_FILE)
    torch.save({
        'X_train_sub': X_tr_sub,
        'y_train_sub': y_tr_sub,
    }, TRAIN_DATA_FILE)
    print(f"   [SAVE] Model saved to: {MODEL_FILE}")
    print(f"   [SAVE] Training data saved to: {TRAIN_DATA_FILE}")


# ============================================================
# SECTION 2: EVALUATION
# ============================================================
def evaluate():
    print(f"\n{'='*50}")
    print(f"  [EVAL] EVALUATION -- {ALGO_NAME}")
    print(f"{'='*50}")

    if not os.path.exists(MODEL_FILE):
        print("   [!] No saved model found! Run --train first.")
        return

    X_train, y_train, X_test, y_test, orig_test = load_data()
    clf = joblib.load(MODEL_FILE)
    saved = torch.load(TRAIN_DATA_FILE)
    X_tr_sub = saved['X_train_sub']

    X_te_sub = X_test[:TEST_SUB]
    y_te_sub = y_test[:TEST_SUB]

    start_time = time.time()
    K_test = compute_kernel_matrix(X_te_sub, X_tr_sub)
    y_pred = clf.predict(K_test)
    elapsed = time.time() - start_time

    show_metrics(ALGO_NAME, y_te_sub.numpy(), y_pred, elapsed)


# ============================================================
# SECTION 3: PREDICTION
# ============================================================
def predict():
    print(f"\n{'='*50}")
    print(f"  [PRED] PREDICTION -- {ALGO_NAME}")
    print(f"{'='*50}")

    if not os.path.exists(MODEL_FILE):
        print("   [!] No saved model found! Run --train first.")
        return

    X_train, y_train, X_test, y_test, orig_test = load_data()
    clf = joblib.load(MODEL_FILE)
    saved = torch.load(TRAIN_DATA_FILE)
    X_tr_sub = saved['X_train_sub']

    X_te_sub = X_test[:TEST_SUB]
    y_te_sub = y_test[:TEST_SUB]

    K_test = compute_kernel_matrix(X_te_sub, X_tr_sub)
    y_pred = clf.predict(K_test)

    n_show = min(10, TEST_SUB)
    orig_sub = orig_test[:TEST_SUB]
    show_predictions(ALGO_NAME, orig_sub, y_te_sub, y_pred, n=n_show)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{ALGO_NAME} - Quantum Support Vector Machine")
    parser.add_argument("--train",    action="store_true", help="Train and save model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set")
    parser.add_argument("--predict",  action="store_true", help="Show predicted images")
    args = parser.parse_args()

    if not (args.train or args.evaluate or args.predict):
        parser.print_help()
    else:
        if args.train:
            train()
        if args.evaluate:
            evaluate()
        if args.predict:
            predict()
