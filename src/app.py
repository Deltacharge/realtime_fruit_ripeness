from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)
# Enable CORS for all routes and origins (for development)
CORS(app, resources={r"/*": {"origins": "*"}})


# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Image transform (SAME AS TRAINING)
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# Load class names
# -----------------------
# Get the project root directory (parent of src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(project_root, "models", "class_names.json")
with open(models_path, "r") as f:
    class_names = json.load(f)

print("FLASK LOADED CLASS ORDER:", class_names)


# -----------------------
# Load model
# -----------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model_path = os.path.join(project_root, "models", "ripeness_model.pth")
model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model = model.to(device)
model.eval()

# -----------------------
# Prediction function
# -----------------------
def predict_image(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return {
        "label": class_names[pred.item()],
        "confidence": round(confidence.item(), 4)
    }

# -----------------------
# Routes
# -----------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fruit Ripeness Prediction API is running 🚀"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(file)
        result = predict_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
