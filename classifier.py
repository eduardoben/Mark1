# .\.venv313\Scripts\Activate.ps1  -> Laptop
# .\.venv\Scripts\Activate.ps1  -> Desktop
# .\training\scripts\Activate.ps1 -> Training models 
# classification_model.py
import torch
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import gc

# Classes labels for the skin lesion classification
class_labels = {
    0: 'actinic keratoses and intraepithelial carcinoma',
    1: 'basal cell carcinoma',
    2: 'benign keratosis-like lesions',
    3: 'dermatofibroma',
    4: 'melanoma',
    5: 'melanocytic nevi',
    6: 'vascular lesions'
}

# Prepare the model
# Using a pre-trained ResNet18 model, modified for 3 channels and 7 classes
n_channels = 3
n_classes = 7

# Check if CUDA is available and set the device accordingly. This is to prioritize GPU usage if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Modify the first convolutional layer to accept 3 channels and the last fully connected layer for 7 classes

# model.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=3, stride=2, padding=3, bias=False)



# Modify the first convolutional layer to match the saved model
model.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, n_classes)

# Load the pre-trained model state dictionary
model.load_state_dict(torch.load(r".\classification model\best_model_state_dict.pth"))
model.to(device)
model.eval()

# Define the preprocessing transformations - resizing, normalization, and conversion to tensor
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to classify an image given its path
# It loads the image, applies preprocessing, and returns the predicted class label
def inference(imagen_path):
    # Load the image and apply preprocessing
    if not imagen_path:
        raise ValueError("Image path cannot be empty")
    
    image = None
    input_tensor = None
    output = None
    predicted_class = None
    
    try:
        image = Image.open(imagen_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        # Ensure the model is in evaluation mode and perform inference
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)
        
        result = class_labels[predicted_class.item()]
        
        return result
    
    finally:
        # Clean up memory after inference
        if image is not None:
            image.close()
            del image
        
        if input_tensor is not None:
            del input_tensor
        
        if output is not None:
            del output
        
        if predicted_class is not None:
            del predicted_class
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()

# Optional: Function to completely unload the model from memory
def cleanup_model():
    """
    Call this function when you're done with all classifications
    to free up model memory completely
    """
    global model
    if 'model' in globals():
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Model cleaned up from memory")