import os
import torch
from torchvision import models
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import re

import torchvision.transforms as transforms
import importlib.util

# Load label mapping from classifier.py
spec = importlib.util.spec_from_file_location("classifier", "classifier.py")
classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classifier)


# Update label mapping
label_map = {
    0: 'actinic keratoses and intraepithelial carcinoma',
    1: 'basal cell carcinoma',
    2: 'benign keratosis-like lesions',
    3: 'dermatofibroma',
    4: 'melanoma',
    5: 'melanocytic nevi',
    6: 'vascular lesions'
}

# Helper to get label from filename
def get_label_from_filename(filename):
    match = re.search(r'\[(\d+)\]', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No label found in filename: {filename}")

# Prepare test dataset
test_dir = "test_images"
test_images = []
test_labels = []
for fname in os.listdir(test_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        test_images.append(os.path.join(test_dir, fname))
        test_labels.append(get_label_from_filename(fname))

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Model loading helper
def load_model(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    num_classes = len(label_map)

    if 'best_model_state_dict.pth' in model_path:
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        n_channels = 3  # Example: Set number of input channels
        model.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif 'resnet18_224_model_state_dict.pth' in model_path:
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        n_channels = 3  # Example: Set number of input channels
        model.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=3, stride=2, padding=3, bias=False)
    else:
        raise ValueError("Unsupported model architecture or missing metadata in state_dict, please check the model file.")

    model.load_state_dict(state_dict)
    model.eval()
    return model

# Evaluate model
def evaluate_model(model):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for img_path, label in zip(test_images, test_labels):
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0)
            outputs = model(input_tensor)
            pred = torch.argmax(outputs, dim=1).item()
            y_true.append(label)
            y_pred.append(pred)
    return y_true, y_pred

# Metrics calculation
def print_metrics(y_true, y_pred, model_name):
    print(f"\n=== Results for {model_name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report (per class):")
    print(classification_report(y_true, y_pred, target_names=[label_map[i] for i in sorted(label_map)]))
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.4f}")
    # UAC: mean of per-class accuracies
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    uac = np.mean(per_class_acc)
    print(f"Unweighted Average Accuracy (UAC): {uac:.4f}")

    # UAC: Weighted Average Accuracy (WAC)
    wac = np.sum(per_class_acc * (cm.sum(axis=1) / cm.sum()))  # Weighted by class support
    print(f"Weighted Average Accuracy (WAC): {wac:.4f}")


# Main loop
model_folder = "classification model"
if not os.path.exists(model_folder):
    raise FileNotFoundError(f"Model folder '{model_folder}' does not exist.")
model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]

for model_file in model_files:
    model_path = os.path.join(model_folder, model_file)
    model = load_model(model_path)
    y_true, y_pred = evaluate_model(model)
    print_metrics(y_true, y_pred, model_file)
