"""
VQML Flask Inference API
================================
Runs local inferences on the drawn interactive demo images using the saved QNN, QCNN, VQC, VQFE, and QSVM models.
Supports both single-shot (/predict) and real-time SSE streaming (/predict-stream) endpoints.
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import base64
import joblib
import json
import time
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt

# Import quantum circuits and configs
from data_utils import load_data, N_QUBITS
from qnn import circuit as qnn_circuit
from qcnn import circuit as qcnn_circuit
from vqc import circuit as vqc_circuit
from vqfe import circuit as vqfe_circuit
from qsvm import qsvm_kernel

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests from the frontend

SAVED_DIR = "saved_models"

# Global model state
models = {
    "pca": None,
    "qsvm_clf": None,
    "qsvm_X_train": None,
    "weights": {},
    "metrics": {},       # Live-computed benchmark metrics per algorithm
    "test_data": None    # (X_test, y_test) for metric computation
}


def setup_models():
    """Load the pre-trained weights and PCA transformation."""
    global models
    print("[*] Initializing VQML Backend...")

    # Load and fit PCA based on the actual training data (to match exactly)
    print("    [1/3] Loading PCA transform parameters...")
    try:
        # Load exactly as during training to get the exact same PCA components.
        # This is a bit slow on startup, but guarantees identical preprocessing.
        from torchvision import datasets
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        idx = (dataset.targets == 2) | (dataset.targets == 7)
        dataset.data = dataset.data[idx]
        dataset.targets = dataset.targets[idx]
        dataset.targets = torch.where(dataset.targets == 2, 0, 1)  # 0=digit2, 1=digit7
        flat_data = dataset.data.float().view(-1, 784).numpy()
        
        pca = PCA(n_components=N_QUBITS)
        pca.fit(flat_data)
        
        pca_transformed = pca.transform(flat_data)
        models["pca_min"] = pca_transformed.min(axis=0)
        models["pca_max"] = pca_transformed.max(axis=0)
        models["pca"] = pca
        
        # Prepare test set for live metric computation
        TRAIN_SIZE = 1000
        TEST_SIZE = 250
        pca_normalized = (pca_transformed - models["pca_min"]) / (models["pca_max"] - models["pca_min"] + 1e-8) * np.pi
        pca_tensor = torch.tensor(pca_normalized, dtype=torch.float32)
        X_test = pca_tensor[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
        y_test = dataset.targets[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
        models["test_data"] = (X_test, y_test)
        print("    [+] PCA Loaded + Test set prepared")
    except Exception as e:
        print(f"    [!] Error initializing PCA: {e}")

    # Load neural network weights
    print("    [2/3] Loading Quantum Neural Network Weights...")
    for algo in ["qnn", "qcnn", "vqc", "vqfe"]:
        path = os.path.join(SAVED_DIR, f"{algo}_weights.pt")
        if os.path.exists(path):
            models["weights"][algo] = torch.load(path)
            print(f"    [+] {algo.upper()} weights loaded")
        else:
            print(f"    [!] Missing {algo.upper()} weights at {path}")

    # Load QSVM
    print("    [3/3] Loading QSVM Model and Training subset...")
    try:
        qsvm_model_path = os.path.join(SAVED_DIR, "qsvm_model.joblib")
        qsvm_train_file = os.path.join(SAVED_DIR, "qsvm_train_data.pt")
        
        if os.path.exists(qsvm_model_path) and os.path.exists(qsvm_train_file):
            models["qsvm_clf"] = joblib.load(qsvm_model_path)
            saved_data = torch.load(qsvm_train_file)
            models["qsvm_X_train"] = saved_data['X_train_sub']
            print("    [+] QSVM Model loaded")
        else:
             print("    [!] Missing QSVM model files")
    except Exception as e:
        print(f"    [!] Error loading QSVM: {e}")

    # Compute live benchmark metrics
    compute_live_metrics()
    print("\n[OK] Backend is ready to accept predictions!\n")


def compute_live_metrics():
    """Run all models on the test set and compute real accuracy/precision/recall/F1."""
    if models["test_data"] is None:
        print("    [!] Test data not available, skipping metric computation")
        return
    
    X_test, y_test = models["test_data"]
    y_true = y_test.numpy()  # 0 = digit 2, 1 = digit 7
    
    circuits = {
        'qnn': qnn_circuit,
        'qcnn': qcnn_circuit,
        'vqc': vqc_circuit,
        'vqfe': vqfe_circuit
    }
    
    print("    [4/4] Computing live benchmark metrics...")
    
    for algo_name, qnode in circuits.items():
        if algo_name not in models["weights"]:
            continue
        weights = models["weights"][algo_name]
        preds = []
        with torch.no_grad():
            for i in range(len(X_test)):
                val = qnode(X_test[i], weights).item()
                preds.append(1 if val > 0 else 0)
        
        y_pred = np.array(preds)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        models["metrics"][algo_name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4)
        }
        print(f"    [+] {algo_name.upper()}: Acc={acc*100:.1f}% Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")
    
    # QSVM metrics
    if models["qsvm_clf"] is not None and models["qsvm_X_train"] is not None:
        try:
            preds = []
            for i in range(len(X_test)):
                digit, _ = predict_qsvm(X_test[i])
                preds.append(1 if digit == 7 else 0)
            
            y_pred = np.array(preds)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            models["metrics"]['qsvm'] = {
                'accuracy': round(acc, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'f1': round(f1, 4)
            }
            print(f"    [+] QSVM: Acc={acc*100:.1f}% Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")
        except Exception as e:
            print(f"    [!] QSVM metric computation failed: {e}")
    
    print(f"    [OK] Metrics computed for {len(models['metrics'])} algorithms")


def preprocess_image(base64_img):
    """
    Decodes the base64 canvas image, resizes it to 28x28 grayscale, 
    and applies PCA identically to the training logic in data_utils.py.
    """
    if "base64," in base64_img:
        base64_img = base64_img.split("base64,")[1]
        
    image_data = base64.b64decode(base64_img)
    image = Image.open(io.BytesIO(image_data))
    
    # Needs to match MNIST: 28x28, grayscale. 
    # Alpha compositing with white background because canvas transparent is black.
    background = Image.new("RGBA", image.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image.convert("RGBA"))
    
    # Convert to grayscale and resize
    grayscale_img = alpha_composite.convert("L").resize((28, 28))
    
    # In MNIST, ink is white (255) and background is black (0).
    # Our canvas has black ink on white background, so we must INVERT it.
    img_array = 255 - np.array(grayscale_img)
    
    # Flatten it
    flat_img = img_array.reshape(1, -1).astype(np.float32)
    
    # Apply PCA
    pca_data = models["pca"].transform(flat_img)
    
    # Normalize to [0, pi]
    pca_normalized = (pca_data - models["pca_min"]) / (models["pca_max"] - models["pca_min"] + 1e-8) * np.pi
    
    return torch.tensor(pca_normalized, dtype=torch.float32).squeeze(0)   # shape: [4]


def predict_qsvm(x_test_tensor):
    """Calculates fidelity kernel matrix for single input against QSVM training subset"""
    if models["qsvm_clf"] is None or models["qsvm_X_train"] is None:
        return 0, 0.5 

    m = len(models["qsvm_X_train"])
    K_test = np.zeros((1, m))
    
    for j, b in enumerate(models["qsvm_X_train"]):
        K_test[0, j] = qsvm_kernel(x_test_tensor, b)
        
    clf = models["qsvm_clf"]
    
    # decision function returns signed distance to hyperplane. 
    # Sigmoid it for confidence
    decision = clf.decision_function(K_test)[0] 
    conf = 1 / (1 + np.exp(-decision * 2)) # Pseudo-sigmoid confidence
    
    pred_label = clf.predict(K_test)[0]
    digit = 7 if pred_label == 1 else 2
    
    # Fix confidence to reflect the predicted digit
    if pred_label == 1:
        final_conf = conf
    else:
        final_conf = 1 - conf 
        
    final_conf = max(0.51, min(0.99, final_conf)) # bound
    return digit, float(final_conf)


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts base64 image and runs inference on all 5 quantum models."""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
        
    try:
        # Preprocess to [0, pi] tensor
        x_tensor = preprocess_image(data['image'])
        results = []
        
        # Helper to process neural network outputs
        def process_nn(name, qnode, weights):
            with torch.no_grad():
                val = qnode(x_tensor, weights).item()
            # output is expval PauliZ [-1, 1]. Map to confidence [0, 1]
            conf = (val + 1) / 2
            pred_digit = 7 if val > 0 else 2
            
            # Bound confidence so it looks like a softmax-ish probability
            if val > 0:
                 final_conf = conf
            else:
                 final_conf = 1 - conf
            final_conf = max(0.51, min(0.99, final_conf))
            
            return {"name": name, "predicted": pred_digit, "confidence": float(final_conf)}

        # Run inferences for each loaded model
        if "qnn" in models["weights"]:
            results.append(process_nn("QNN", qnn_circuit, models["weights"]["qnn"]))
        if "qcnn" in models["weights"]:
            results.append(process_nn("QCNN", qcnn_circuit, models["weights"]["qcnn"]))
        if "vqc" in models["weights"]:
            results.append(process_nn("VQC", vqc_circuit, models["weights"]["vqc"]))
        if "vqfe" in models["weights"]:
            results.append(process_nn("VQFE", vqfe_circuit, models["weights"]["vqfe"]))
            
        if models["qsvm_clf"]:
            qsvm_digit, qsvm_conf = predict_qsvm(x_tensor)
            results.append({"name": "QSVM", "predicted": qsvm_digit, "confidence": qsvm_conf})

        # Weighted consensus: only trust models with >75% benchmark accuracy
        accurate_models = {'QCNN', 'VQC', 'QSVM'}  # QNN (52.8%) and VQFE (57.6%) excluded
        trusted = [r for r in results if r['name'] in accurate_models]
        if trusted:
            votes_7 = sum(1 for r in trusted if r['predicted'] == 7)
            votes_2 = sum(1 for r in trusted if r['predicted'] == 2)
            consensus = 7 if votes_7 > votes_2 else 2
        else:
            # Fallback to all models if none are trusted
            votes_7 = sum(1 for r in results if r['predicted'] == 7)
            votes_2 = sum(1 for r in results if r['predicted'] == 2)
            consensus = 7 if votes_7 > votes_2 else 2

        response = {
            "success": True,
            "consensus": consensus,
            "algorithms": results,
            "metrics": models["metrics"]
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def fig_to_base64(fig, dpi=100):
    """Convert a matplotlib figure to a Base64-encoded PNG data URL."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none', transparent=False)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def generate_preprocessed_image(base64_img):
    """Generate a visualization of the preprocessed 28x28 grayscale input."""
    if "base64," in base64_img:
        base64_img = base64_img.split("base64,")[1]
    image_data = base64.b64decode(base64_img)
    image = Image.open(io.BytesIO(image_data))

    background = Image.new("RGBA", image.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image.convert("RGBA"))
    grayscale_img = alpha_composite.convert("L").resize((28, 28))
    img_array = 255 - np.array(grayscale_img)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img_array, cmap='magma', interpolation='nearest')
    ax.set_title('Model Input (28×28)', color='#e0e0e0', fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#333355')
    fig.patch.set_facecolor('#1a1a2e')
    return fig_to_base64(fig)


def generate_pca_chart(x_tensor):
    """Generate a bar chart of the 4 PCA component values."""
    values = x_tensor.numpy()
    colors = ['#5b7fa6', '#8e7ab5', '#c47a8e', '#6aab8e']
    labels = [f'PC{i+1}' for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(4, 2.5))
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor='none', alpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', color='#e0e0e0',
                fontsize=9, fontweight='bold')

    ax.set_title('PCA Components → Qubit Angles', color='#e0e0e0', fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel('Value (0 to π)', color='#9a9490', fontsize=9)
    ax.set_ylim(0, max(values) * 1.3 + 0.1)
    ax.tick_params(colors='#9a9490', labelsize=9)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.spines['bottom'].set_color('#333355')
    ax.spines['left'].set_color('#333355')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig_to_base64(fig)


def generate_result_badge(name, predicted, confidence):
    """Generate a small result badge image for an algorithm's prediction."""
    color_map = {
        'QNN': '#c47a8e', 'QCNN': '#5b7fa6', 'VQC': '#8e7ab5',
        'VQFE': '#c9a95a', 'QSVM': '#6aab8e'
    }
    color = color_map.get(name, '#5b7fa6')

    fig, ax = plt.subplots(figsize=(3, 1.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_axis_off()
    fig.patch.set_facecolor('#1a1a2e')

    # Algorithm name
    ax.text(0.5, 3.0, name, color=color, fontsize=14, fontweight='bold', va='center')
    # Predicted digit
    ax.text(0.5, 1.5, f'→ Digit {predicted}', color='#e0e0e0', fontsize=18, fontweight='bold', va='center')
    # Confidence bar background
    ax.barh(0.3, 10, height=0.5, color='#333355', left=0)
    # Confidence bar fill
    ax.barh(0.3, confidence * 10, height=0.5, color=color, left=0, alpha=0.85)
    ax.text(confidence * 10 + 0.2, 0.3, f'{confidence*100:.1f}%', color='#e0e0e0',
            fontsize=9, fontweight='bold', va='center')

    return fig_to_base64(fig, dpi=80)


# ============================================================
# SSE STREAMING ENDPOINT
# ============================================================

def sse_event(event_type, data_dict):
    """Format a Server-Sent Event string."""
    return f"event: {event_type}\ndata: {json.dumps(data_dict)}\n\n"


@app.route("/predict-stream", methods=["POST"])
def predict_stream():
    """SSE endpoint: streams processing steps and visualizations in real-time."""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    raw_image = data['image']

    def generate():
        try:
            # Step 1: Preprocessed input visualization
            preproc_img = generate_preprocessed_image(raw_image)
            yield sse_event('step', {
                'step': 'preprocessing',
                'image': preproc_img,
                'label': 'Preprocessed Input (28×28)'
            })
            time.sleep(0.3)  # Small delay for visual streaming effect

            # Step 2: PCA + angle encoding
            x_tensor = preprocess_image(raw_image)
            pca_img = generate_pca_chart(x_tensor)
            yield sse_event('step', {
                'step': 'pca',
                'image': pca_img,
                'label': 'PCA Feature Encoding'
            })
            time.sleep(0.3)

            # Step 3: Per-algorithm inference (streamed one at a time)
            all_results = []

            def run_nn(name, qnode, weights):
                with torch.no_grad():
                    val = qnode(x_tensor, weights).item()
                conf = (val + 1) / 2
                pred_digit = 7 if val > 0 else 2
                if val > 0:
                    final_conf = conf
                else:
                    final_conf = 1 - conf
                final_conf = max(0.51, min(0.99, final_conf))
                return pred_digit, float(final_conf)

            algo_order = [
                ("QNN", "qnn", lambda: run_nn("QNN", qnn_circuit, models["weights"]["qnn"]) if "qnn" in models["weights"] else None),
                ("QCNN", "qcnn", lambda: run_nn("QCNN", qcnn_circuit, models["weights"]["qcnn"]) if "qcnn" in models["weights"] else None),
                ("VQC", "vqc", lambda: run_nn("VQC", vqc_circuit, models["weights"]["vqc"]) if "vqc" in models["weights"] else None),
                ("VQFE", "vqfe", lambda: run_nn("VQFE", vqfe_circuit, models["weights"]["vqfe"]) if "vqfe" in models["weights"] else None),
                ("QSVM", "qsvm", lambda: predict_qsvm(x_tensor) if models["qsvm_clf"] else None),
            ]

            for display_name, key, run_fn in algo_order:
                result = run_fn()
                if result is None:
                    continue
                pred_digit, conf = result
                badge_img = generate_result_badge(display_name, pred_digit, conf)
                algo_result = {
                    'name': display_name,
                    'predicted': pred_digit,
                    'confidence': conf,
                    'image': badge_img
                }
                all_results.append(algo_result)
                yield sse_event('result', algo_result)
                time.sleep(0.4)  # Staggered streaming effect

            # Step 4: Weighted consensus (only trust accurate models)
            accurate_models = {'QCNN', 'VQC', 'QSVM'}
            trusted = [r for r in all_results if r['name'] in accurate_models]
            if trusted:
                votes_7 = sum(1 for r in trusted if r['predicted'] == 7)
                votes_2 = sum(1 for r in trusted if r['predicted'] == 2)
                consensus = 7 if votes_7 > votes_2 else 2
            else:
                votes_7 = sum(1 for r in all_results if r['predicted'] == 7)
                votes_2 = sum(1 for r in all_results if r['predicted'] == 2)
                consensus = 7 if votes_7 > votes_2 else 2

            yield sse_event('done', {
                'consensus': consensus,
                'algorithms': [{'name': r['name'], 'predicted': r['predicted'], 'confidence': r['confidence']} for r in all_results],
                'metrics': models['metrics']
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield sse_event('error', {'error': str(e)})

    return Response(generate(), content_type='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == "__main__":
    setup_models()
    # Run locally on port 5000
    app.run(host="127.0.0.1", port=5000)
