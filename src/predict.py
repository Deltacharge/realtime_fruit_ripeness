import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import os

# ------------------
# Device
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------
# SAME transform as training
# ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------
# Load model
# ------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("models/ripeness_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ------------------
# Class labels (IMPORTANT)
# ------------------
class_names = ["fresh", "rotten", "okay"]

# ------------------
# Predict single image
# ------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()

# ------------------
# TEST on one image
# ------------------
if __name__ == "__main__":
    test_image_path = "data/test/rotten/rotated_by_15_Screen Shot 2018-06-07 at 2.21.09 PM.png"  # CHANGE THIS

    if not os.path.exists(test_image_path):
        print("Image not found:", test_image_path)
    else:
        label, confidence = predict_image(test_image_path)
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}")