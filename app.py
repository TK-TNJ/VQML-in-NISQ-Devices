"""
VQML Flask Inference API
================================
Runs local inferences on the drawn interactive demo images using the saved QNN, QCNN, VQC, VQFE, and QSVM models.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import base64
import joblib
from sklearn.decomposition import PCA
import os

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
    "weights": {}
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
        flat_data = dataset.data.float().view(-1, 784).numpy()
        
        pca = PCA(n_components=N_QUBITS)
        pca.fit(flat_data)
        
        pca_transformed = pca.transform(flat_data)
        models["pca_min"] = pca_transformed.min(axis=0)
        models["pca_max"] = pca_transformed.max(axis=0)
        models["pca"] = pca
        print("    [+] PCA Loaded")
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

    print("\n[OK] Backend is ready to accept predictions!\n")


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

        # Calculate a consensus prediction (majority vote among models)
        votes_7 = sum(1 for r in results if r['predicted'] == 7)
        votes_2 = sum(1 for r in results if r['predicted'] == 2)
        consensus = 7 if votes_7 > votes_2 else 2

        response = {
            "success": True,
            "consensus": consensus,
            "algorithms": results
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    setup_models()
    # Run locally on port 5000
    app.run(host="127.0.0.1", port=5000)

