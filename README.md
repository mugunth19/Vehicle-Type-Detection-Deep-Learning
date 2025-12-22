# Vehicle Type Classification Project (MobileNetV2 - ONNX Export)

## Overview
This project aims to develop a vehicle type classification model capable of distinguishing between `hatchback`, `motorcycle`, `pickup`, `sedan`, and `suv` vehicles. The solution utilizes a pre-trained MobileNetV2 backbone, fine-tuned on a custom dataset, and then exported to ONNX format for efficient and portable inference.

## Project Structure

- `vehicle-type/`: Original dataset directory after unzipping.
- `vehicle_data_split/`: Dataset split into `train`, `val`, and `test` directories.
- `vehicleclassifier_WDA_32_0.899.pth`: Best performing PyTorch model checkpoint.
- `vehicle_identifier_mobilenet_v2.onnx`: The final model exported in ONNX format.

## Dataset

The dataset consists of images of various vehicle types. It was obtained from a ZIP file (`r7bthvstxw-2.zip`), unzipped, and renamed to `vehicle-type/`. The dataset contains the following classes:

- `hatchback`
- `motorcycle`
- `pickup`
- `sedan`
- `suv`

### Class Distribution
The initial analysis revealed an imbalance in the class distribution. For instance, 'pickup' and 'sedan' classes have significantly more samples than 'motorcycle' or 'suv'. To address this, a **weighted cross-entropy loss function** was employed during training.

| Vehicle Class | Number of Samples |
|---------------|-------------------|
| hatchback     | 181               |
| motorcycle    | 122               |
| pickup        | 478               |
| sedan         | 400               |
| suv           | 129               |

## Data Preparation and Augmentation

### Custom Dataset Class
A `VehicleDataset` class was implemented to efficiently load images and their corresponding labels from disk. It handles reading image files, converting them to RGB, and applying transformations.

### Data Splitting
The dataset was split into training, validation, and test sets using a ratio of 80% for training, 10% for validation, and 10% for testing. This ensures a robust evaluation of the model's performance on unseen data.

### Transformations
To enhance model generalization and prevent overfitting, extensive data augmentation was applied to the training set:

-   **Random Horizontal Flip**: Flips images horizontally with a 50% probability, as the orientation of a vehicle does not change its type.
-   **Random Rotation**: Rotates images by up to 15 degrees, useful for real-world scenarios where camera angles might vary.
-   **Color Jitter**: Adjusts brightness, contrast, and saturation (each by 20%), helping the model become robust to different lighting conditions and vehicle colors.
-   **Random Resized Crop**: Crops a random portion of the image and resizes it to the target `input_size`, simulating varying distances and perspectives of vehicles.

The validation set only underwent resizing and normalization.

## Model Architecture

The model is based on the **MobileNetV2** architecture, pre-trained on ImageNet. The pre-trained weights are frozen to leverage learned features, and a custom head is added for classification:

-   **Base Model**: `mobilenet_v2` (features frozen).
-   **Global Average Pooling**: `nn.AdaptiveAvgPool2d((1, 1))` to reduce spatial dimensions.
-   **Dropout Layer**: `nn.Dropout(p=0.2)` applied to the flattened features to reduce overfitting.
-   **Output Layer**: `nn.Linear(1280, 5)` to classify into the 5 vehicle types.

## Training Details

-   **Loss Function**: `nn.CrossEntropyLoss` with class weights to counteract dataset imbalance.
-   **Optimizer**: Adam optimizer.
-   **Learning Rate**: 0.01 (selected after hyperparameter tuning).
-   **Dropout Probability**: 0.2 (selected after hyperparameter tuning).
-   **Epochs**: 50.
-   **Device**: Training performed on GPU if available, otherwise CPU.

## Hyperparameter Tuning

Initial experiments were conducted to determine optimal hyperparameters:

1.  **Learning Rate**: Tested `0.001`, `0.01`, `0.1`. A learning rate of `0.01` yielded the best performance (Validation Accuracy of 93.80% in one run, but with significant loss fluctuations).
2.  **Inner Layer Size**: Experimented with adding a new inner linear layer (sizes `100, 250, 500, 750, 1000`). This did not improve performance; in fact, it led to much higher loss and lower accuracy, indicating that the original simple head was sufficient.
3.  **Dropout**: Tested dropout probabilities `0.2`, `0.4`, `0.5`, `0.7`. A dropout of `0.2` produced the most stable and highest validation accuracy with data augmentation.

The chosen configuration (LR=0.01, Dropout=0.2, no inner layer, with data augmentation) provided the best balance between training performance and generalization.

## Performance

The final model, trained with data augmentation, a learning rate of 0.01, and a dropout of 0.2, achieved a **validation accuracy of 89.9%** (checkpoint saved at epoch 32). While a non-augmented model occasionally showed slightly higher peak accuracy (e.g., 92%), the augmented model demonstrated superior generalization capabilities and more consistent performance across epochs due to its exposure to a wider variety of image transformations.

## ONNX Export

To facilitate deployment and efficient inference in various environments, the trained PyTorch model was exported to the ONNX (Open Neural Network Exchange) format. The ONNX model preserves the architecture and weights, allowing it to be run with ONNX Runtime or other ONNX-compatible engines.

-   **ONNX Model File**: `vehicle_identifier_mobilenet_v2.onnx`

## How to Use the ONNX Model (Python Example for Inference)

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

# Define preprocessing transforms (must match training transforms - specifically validation transforms)
input_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Example image path (replace with an actual image path)
example_image_path = "path/to/your/image.jpg"

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

# Define class labels (must match the order used during training)
class_labels = ['hatchback', 'motorcycle', 'pickup', 'sedan', 'suv']

# Get the predicted class index and probability
predicted_class_idx = np.argmax(probabilities, axis=1)[0]
predicted_class_label = class_labels[predicted_class_idx]
confidence = probabilities[0, predicted_class_idx]

print(f"Predicted Vehicle Type: {predicted_class_label}")
print(f"Confidence: {confidence:.4f}")
```
