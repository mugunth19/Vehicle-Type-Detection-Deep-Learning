

get_ipython().system('wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/r7bthvstxw-2.zip')



get_ipython().system('unzip r7bthvstxw-2.zip  && mv r7bthvstxw-2 vehicle-type')


import torch
from PIL import Image
import numpy as np
import torchvision.models as models
from torchvision import transforms


import splitfolders

# Path to your current unsplit folder
input_folder = 'vehicle-type'

# Split with a ratio: 80% Train, 10% Val, 10% Test
# Seed ensures the split is reproducible every time you run it
splitfolders.ratio(input_folder, output="vehicle_data_split",
                   seed=1337, ratio=(.8, .1, .1))


# In[ ]:


import os
from torch.utils.data import Dataset
from PIL import Image

# Define a custom Dataset class for loading vehicle images
class VehicleDataset(Dataset):
    # Initialize the dataset
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir  # Root directory containing class folders
        self.transform = transform  # Transformations to apply to images (e.g., resizing, normalization)
        self.image_paths = []     # List to store full paths to all images
        self.labels = []          # List to store integer labels corresponding to each image

        # Get class names from subdirectories and sort them for consistent indexing
        self.classes = sorted(os.listdir(data_dir))
        # Create a mapping from class name (string) to integer index
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Populate image_paths and labels lists
        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)  # Path to the current class's directory
            # Iterate through all image files in the class directory
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))  # Add image path
                self.labels.append(self.class_to_idx[label_name])           # Add corresponding integer label

    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.image_paths)

    # Return a single sample (image and its label) given an index
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  # Get the path of the image at the given index
        image = Image.open(img_path).convert('RGB')  # Open and convert image to RGB format
        label = self.labels[idx]          # Get the label for the image

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label


# In[ ]:


import torch.nn as nn

class VehicleClassifierModel(nn.Module):
    def __init__(self, num_classes=5,dropout_p=0.0):
        super(VehicleClassifierModel, self).__init__()

        # Load pre-trained MobileNetV2
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Remove original classifier
        self.base_model.classifier = nn.Identity()

        # Add custom layers
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # dropout layer
        self.dropout = nn.Dropout(p=dropout_p)

        self.output_layer = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        # removed the previously added inner layer
        # added dropout layer
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


# In[ ]:


def  train_and_val(model, optimizer, train_loader, val_loader, criterion, num_epochs, device) :
  best_val_accuracy = 0.0  # Initialize variable to track the best validation accuracy

  for epoch in range(num_epochs):
      # Training phase
      model.train()  # Set the model to training mode
      running_loss = 0.0
      correct = 0
      total = 0

      # Iterate over the training data
      for inputs, labels in train_loader:
          # Move data to the specified device (GPU or CPU)
          inputs, labels = inputs.to(device), labels.to(device)

          # Zero the parameter gradients to prevent accumulation
          optimizer.zero_grad()
          # Forward pass
          outputs = model(inputs)
          # Calculate the loss
          loss = criterion(outputs, labels)
          # Backward pass and optimize
          loss.backward()
          optimizer.step()

          # Accumulate training loss
          running_loss += loss.item()
          # Get predictions
          _, predicted = torch.max(outputs.data, 1)
          # Update total and correct predictions
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

      # Calculate average training loss and accuracy
      train_loss = running_loss / len(train_loader)
      train_acc = correct / total

      # Validation phase
      model.eval()  # Set the model to evaluation mode
      val_loss = 0.0
      val_correct = 0
      val_total = 0

      # Disable gradient calculation for validation
      with torch.no_grad():
          # Iterate over the validation data
          for inputs, labels in val_loader:
              # Move data to the specified device (GPU or CPU)
              inputs, labels = inputs.to(device), labels.to(device)
              # Forward pass
              outputs = model(inputs)
              # Calculate the loss
              loss = criterion(outputs, labels)

              # Accumulate validation loss
              val_loss += loss.item()
              # Get predictions
              _, predicted = torch.max(outputs.data, 1)
              # Update total and correct predictions
              val_total += labels.size(0)
              val_correct += (predicted == labels).sum().item()

      # Calculate average validation loss and accuracy
      val_loss /= len(val_loader)
      val_acc = val_correct / val_total

      # Print epoch results
      print(f'Epoch {epoch+1}/{num_epochs}')
      print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
      print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

      if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            checkpoint_path = f'vehicleclassifier_WDA_{epoch+1:02d}_{val_acc:.3f}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')


# In[ ]:


def make_model(learning_rate=0.01, class_weights=None,dropout_p=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VehicleClassifierModel(num_classes=5,dropout_p=dropout_p)
    model.to(device) # Move the model to the correct device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 1. Define the loss function with your calculated weights
    # If you are using a GPU, make sure the weights are on the same device
    if class_weights is not None:
      class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return model, optimizer, criterion


# In[ ]:


input_size = 224

# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

val_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# In[ ]:


train_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),

    # 1. Horizontal Flip: A car is still a car if viewed from the other side.
    transforms.RandomHorizontalFlip(p=0.5),

    # 2. Random Rotation: Handles cars parked at slight angles or tilted cameras.
    transforms.RandomRotation(15),

    # 3. Color Jitter: Simulates different lighting (sunny vs. cloudy) and car colors.
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    # 4. Random Resized Crop: Simulates the car being closer or further from the camera.
    transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),

    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# In[ ]:


from torch.utils.data import DataLoader

train_dataset = VehicleDataset(
    data_dir='./vehicle_data_split/train',
    transform=train_transforms
)

val_dataset = VehicleDataset(
    data_dir='./vehicle_data_split/val',
    transform=val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# In[ ]:


# Based on our experiments :
learning_rate = 0.01
dropout = 0.02
epochs = 50 # since we have added dropout, increasing epochs
#No inner layer needed


# In[ ]:


import torch
import os

# Path to your main training directory
DATA_DIR = 'vehicle-type'

# Dynamically get counts from folders
class_names = sorted(os.listdir(DATA_DIR))
counts = [len(os.listdir(os.path.join(DATA_DIR, c))) for c in class_names]

total = sum(counts)
num_classes = len(counts)

# Calculate weights: Total / (num_classes * class_count)
weights = [total / (num_classes * c) for c in counts]

# Convert to a FloatTensor for PyTorch
class_weights = torch.FloatTensor(weights)

print(f"Classes: {class_names}")
print(f"Counts: {counts}")
print(f"Calculated Weights: {weights}")


# In[ ]:


model, optimizer, criterion = make_model(learning_rate=learning_rate,dropout_p=dropout,class_weights=class_weights)
train_and_val(model, optimizer, train_loader, val_loader, criterion, epochs, device)

