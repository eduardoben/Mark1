#  .\.venv313\Scripts\Activate.ps1    
import torch
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

# Define the dataset-specific details
n_channels = 3  # For RGB images (e.g., MedMNIST or your dataset)
n_classes = 7  # Replace with the actual number of classes in your task

# Load the ResNet-18 model with pre-trained weights (use the 'weights' argument)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Using pre-trained weights from ImageNet


# Modify the first convolutional layer if necessary (based on your training setup)
model.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=3, stride=2, padding=3, bias=False)

# Modify the fully connected layer (output layer)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, n_classes)

# Load the trained weights
model.load_state_dict(torch.load(r".\classification model\resnet18_224_model_state_dict.pth"))


# Set the model to evaluation mode
model.eval()

# Define the image transformations (same as during training)
# preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load and preprocess the image
img_path = r".\test_images\381_[6].png"  # Change this to your image path

input_image = Image.open(img_path)
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

# Move the model and input to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
input_batch = input_batch.to(device)

# Perform inference (predict)
with torch.no_grad():
    output = model(input_batch)

# Get the predicted class
_, predicted_class = torch.max(output, 1)

# Print the predicted class
print(f"Predicted class: {predicted_class.item()}")

# Mapping the predicted class to the label
class_labels = {
    0: 'actinic keratoses and intraepithelial carcinoma',
    1: 'basal cell carcinoma',
    2: 'benign keratosis-like lesions',
    3: 'dermatofibroma',
    4: 'melanoma',
    5: 'melanocytic nevi',
    6: 'vascular lesions'
}

# Print the predicted class name
print(f"Predicted class: {class_labels[predicted_class.item()]}")
