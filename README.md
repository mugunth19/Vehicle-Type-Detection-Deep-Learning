# Vehicle Type Classifier (MobileNetV2 - ONNX Export)

## Overview
This repository contains a vehicle type classification model, exported to ONNX format for efficient inference. The model is built using a pre-trained MobileNetV2 backbone and fine-tuned for classifying different types of vehicles: **hatchback, motorcycle, pickup, sedan, and suv**.

## Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Head**: A global average pooling layer followed by a linear output layer (1280 features to 5 classes).
- **Dropout**: A dropout layer with a probability of 0.2 is applied before the final classification layer to prevent overfitting.

## Training Details
- **Dataset**: A custom dataset of vehicle images, split into training, validation, and test sets.
- **Loss Function**: Weighted Cross-Entropy Loss was used to address class imbalance within the dataset.
- **Optimizer**: Adam optimizer with a learning rate of 0.01.
- **Epochs**: 50 epochs.
- **Data Augmentation**: Extensive data augmentation techniques were applied to the training data to improve generalization, including:
    - Random Horizontal Flip
    - Random Rotation (up to 15 degrees)
    - Color Jitter (brightness, contrast, saturation)
    - Random Resized Crop

## Performance
During training, the model achieved a **validation accuracy of approximately 89.9%** (recorded at epoch 32 with the best checkpoint: `vehicleclassifier_WDA_32_0.899.pth`). The use of data augmentation helped the model achieve better generalization compared to models trained without it.

## Input and Output
- **Input**: A 3-channel RGB image of shape `(batch_size, 3, 224, 224)`. The input image should be normalized using the following ImageNet statistics:
    - Mean: `[0.485, 0.456, 0.406]`
    - Standard Deviation: `[0.229, 0.224, 0.225]`
- **Output**: A tensor of shape `(batch_size, 5)` representing the raw logits for each of the 5 vehicle classes. To get probabilities, apply a Softmax function.

## How to Use the ONNX Model (Python Example)

### 1. Install ONNX Runtime
```bash
pip install onnxruntime
```

### 2. Load and Run Inference
```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Define the path to your ONNX model
onnx_model_path = "vehicle_identifier_mobilenet_v2.onnx"

# Load the ONNX model
sess = ort.InferenceSession(onnx_model_path)

# Get input and output names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Define preprocessing transforms (must match training transforms)
input_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Example image path
example_image_path = "path/to/your/image.jpg" # Replace with an actual image path

# Load and preprocess the image
img = Image.open(example_image_path).convert('RGB')
img_tensor = preprocess(img)

# Add a batch dimension
input_batch = img_tensor.unsqueeze(0).numpy()

# Run inference
outputs = sess.run([output_name], {input_name: input_batch})

# Get the raw logits
logits = outputs[0]

# Apply softmax to get probabilities (optional)
probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

# Define class labels
class_labels = ['hatchback', 'motorcycle', 'pickup', 'sedan', 'suv']

# Get the predicted class index and probability
predicted_class_idx = np.argmax(probabilities, axis=1)[0]
predicted_class_label = class_labels[predicted_class_idx]
confidence = probabilities[0, predicted_class_idx]

print(f"Predicted Vehicle Type: {predicted_class_label}")
print(f"Confidence: {confidence:.4f}")
```

## Files
- `vehicle_identifier_mobilenet_v2.onnx`: The exported ONNX model.
- `vehicleclassifier_WDA_32_0.899.pth`: The PyTorch checkpoint saved during training.
