"""
QCNN - Quantum Convolutional Neural Network
=============================================
Binary MNIST classification (Digit 2 vs 7) using PCA feature reduction,
AngleEmbedding + Conv-like (RX/RY + CNOT) + Pooling-like (Rot).

Usage:
    python qcnn.py --train       # Train & save weights
    python qcnn.py --evaluate    # Load weights & show metrics
    python qcnn.py --predict     # Load weights & show predicted images
"""

import argparse
import torch
import pennylane as qml
import numpy as np
import time
import os

from data_utils import (
    load_data, dev, N_QUBITS, EPOCHS, SAVED_DIR,
    show_metrics, show_predictions
)

ALGO_NAME   = "QCNN"
WEIGHT_FILE = os.path.join(SAVED_DIR, "qcnn_weights.pt")
WEIGHT_SHAPE = (2, 3)


# ============================================================
# CIRCUIT DEFINITION
# ============================================================
@qml.qnode(dev)
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    for i in range(0, N_QUBITS, 2):
        qml.RX(weights[0][0], wires=i)
        qml.RY(weights[0][1], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
    qml.Rot(*weights[1], wires=0)
    return qml.expval(qml.PauliZ(0))


# ============================================================
# SECTION 1: TRAINING
# ============================================================
def train():
    print(f"\n{'='*50}")
    print(f"  [TRAIN] TRAINING -- {ALGO_NAME}")
    print(f"{'='*50}")

    X_train, y_train, X_test, y_test, orig_test = load_data()

    weights = torch.randn(WEIGHT_SHAPE, requires_grad=True)
    opt = torch.optim.Adam([weights], lr=0.1)

    start_time = time.time()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        preds = torch.stack([circuit(x, weights) for x in X_train])
        targets = (y_train * 2) - 1
        loss = torch.mean((preds - targets) ** 2)
        loss.backward()
        opt.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:>3}/{EPOCHS} | Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    print(f"\n   [OK] Training complete in {elapsed:.1f}s")

    torch.save(weights.detach(), WEIGHT_FILE)
    print(f"   [SAVE] Weights saved to: {WEIGHT_FILE}")


# ============================================================
# SECTION 2: EVALUATION
# ============================================================
def evaluate():
    print(f"\n{'='*50}")
    print(f"  [EVAL] EVALUATION -- {ALGO_NAME}")
    print(f"{'='*50}")

    if not os.path.exists(WEIGHT_FILE):
        print("   [!] No saved weights found! Run --train first.")
        return

    X_train, y_train, X_test, y_test, orig_test = load_data()
    weights = torch.load(WEIGHT_FILE)

    start_time = time.time()
    with torch.no_grad():
        preds_raw = torch.stack([circuit(x, weights) for x in X_test])
        y_pred = (preds_raw > 0).int().numpy()
    elapsed = time.time() - start_time

    show_metrics(ALGO_NAME, y_test.numpy(), y_pred, elapsed)


# ============================================================
# SECTION 3: PREDICTION
# ============================================================
def predict():
    print(f"\n{'='*50}")
    print(f"  [PRED] PREDICTION -- {ALGO_NAME}")
    print(f"{'='*50}")

    if not os.path.exists(WEIGHT_FILE):
        print("   [!] No saved weights found! Run --train first.")
        return

    X_train, y_train, X_test, y_test, orig_test = load_data()
    weights = torch.load(WEIGHT_FILE)

    with torch.no_grad():
        preds_raw = torch.stack([circuit(x, weights) for x in X_test])
        y_pred = (preds_raw > 0).int().numpy()

    show_predictions(ALGO_NAME, orig_test, y_test, y_pred, n=10)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{ALGO_NAME} - Quantum Convolutional Neural Network")
    parser.add_argument("--train",    action="store_true", help="Train and save weights")
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
