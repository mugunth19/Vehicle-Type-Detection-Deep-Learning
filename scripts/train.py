import os
import subprocess
import zipfile
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models
from torchvision import transforms
import splitfolders

def setup_data():
    """Download and split dataset"""
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/r7bthvstxw-2.zip'
    zip_file = 'r7bthvstxw-2.zip'
    
    if not os.path.exists('vehicle-type'):
        # Download
        print("Downloading dataset...")
        response = requests.get(url)
        with open(zip_file, 'wb') as f:
            f.write(response.content)
        
        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        # Rename
        os.rename('r7bthvstxw-2', 'vehicle-type')
        os.remove(zip_file)
    
    if not os.path.exists('vehicle_data_split'):
        print("Splitting dataset...")
        splitfolders.ratio('vehicle-type', output="vehicle_data_split",
                           seed=1337, ratio=(.8, .1, .1))


class VehicleDataset(Dataset):
    """Custom Dataset for vehicle images"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(self.class_to_idx[label_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label



class VehicleClassifierModel(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.0):
        super().__init__()
        
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.base_model.classifier = nn.Identity()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.output_layer = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


def train_and_val(model, optimizer, train_loader, val_loader, criterion, num_epochs, device):
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            checkpoint_path = f'vehicleclassifier_WDA_{epoch+1:02d}_{val_acc:.3f}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')


def make_model(learning_rate=0.01, class_weights=None, dropout_p=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VehicleClassifierModel(num_classes=5, dropout_p=dropout_p)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    return model, optimizer, criterion



def get_transforms():
    """Get training and validation transforms"""
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transforms, val_transforms


def create_data_loaders(batch_size=32):
    """Create training and validation data loaders"""
    train_transforms, val_transforms = get_transforms()
    
    train_dataset = VehicleDataset('./vehicle_data_split/train', transform=train_transforms)
    val_dataset = VehicleDataset('./vehicle_data_split/val', transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def calculate_class_weights(data_dir='vehicle-type'):
    """Calculate class weights for imbalanced dataset"""
    class_names = sorted(os.listdir(data_dir))
    counts = [len(os.listdir(os.path.join(data_dir, c))) for c in class_names]
    
    total = sum(counts)
    num_classes = len(counts)
    weights = [total / (num_classes * c) for c in counts]
    
    print(f"Classes: {class_names}")
    print(f"Counts: {counts}")
    print(f"Weights: {weights}")
    
    return torch.FloatTensor(weights)


def main():
    """Main training function"""
    learning_rate = 0.01
    dropout = 0.02
    epochs = 50
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_data_loaders()
    
    class_weights = calculate_class_weights()
    model, optimizer, criterion = make_model(learning_rate=learning_rate, 
                                           dropout_p=dropout, 
                                           class_weights=class_weights)
    
    train_and_val(model, optimizer, train_loader, val_loader, criterion, epochs, device)


if __name__ == "__main__":
    setup_data()
    main()
