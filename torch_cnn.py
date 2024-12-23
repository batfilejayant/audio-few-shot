import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_fscore_support, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Check for GPU availability
print("Checking for GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths for spectrograms
emergency_dir = "/mnt/c/Users/jayant-few-shot/Few_shot/new-sounds/emergency-sounds/emergency_spectrograms"
normal_dir = "/mnt/c/Users/jayant-few-shot/Few_shot/new-sounds/normal-sounds/spectogram_normal_sounds"
results_dir = "/mnt/c/Users/jayant-few-shot/Few_shot/neural-networks-and-results"

# Ensure results directory exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Output CSV file paths
emergency_csv = "emergency_sounds_labels.csv"
normal_csv = "normal_sounds_labels.csv"

# Function to verify and fix images
def verify_and_fix_images(directory):
    for file in os.listdir(directory):
        if file.endswith(".png"):
            file_path = os.path.join(directory, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Check for issues
            except (IOError, SyntaxError):
                print(f"Corrupted file detected and removed: {file_path}")
                os.remove(file_path)

# Verify and fix images in both directories
verify_and_fix_images(emergency_dir)
verify_and_fix_images(normal_dir)

# Function to generate CSV
def generate_csv(directory, label, output_csv):
    data = []
    for i, file in enumerate(os.listdir(directory)):
        if file.endswith(".png") and i < 100:  # Take only 100 samples per class
            file_path = os.path.join(directory, file)
            data.append({"file_path": file_path, "label": label})
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV file created: {output_csv}")

# Generate CSV files for emergency and normal sounds
generate_csv(emergency_dir, "emergency", emergency_csv)
generate_csv(normal_dir, "normal", normal_csv)

# Load and combine datasets
emergency_df = pd.read_csv(emergency_csv)
normal_df = pd.read_csv(normal_csv)
full_df = pd.concat([emergency_df, normal_df], ignore_index=True)

# Split dataset into train, validation, and test sets
train, test = train_test_split(full_df, test_size=0.2, stratify=full_df['label'], random_state=42)
val, test = train_test_split(test, test_size=0.5, stratify=test['label'], random_state=42)

# Save splits to CSV files
train.to_csv("train_labels.csv", index=False)
val.to_csv("val_labels.csv", index=False)
test.to_csv("test_labels.csv", index=False)
print("Dataset splits saved: train_labels.csv, val_labels.csv, test_labels.csv")

# Custom Dataset class
class SpectrogramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['file_path']).convert('RGB')
        label = 1 if row['label'] == 'emergency' else 0
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets and DataLoaders
train_dataset = SpectrogramDataset(train, transform=transform)
val_dataset = SpectrogramDataset(val, transform=transform)
test_dataset = SpectrogramDataset(test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Define the CNN model
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
model = nn.DataParallel(model)  # Enable multi-GPU support
model = model.to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete")

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10)

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
    return y_true, y_pred_binary

# Get predictions
y_true, y_pred = evaluate_model(model, test_loader)

# Classification report
class_report = classification_report(y_true, y_pred, target_names=["normal", "emergency"])
print(class_report)

# Save classification report
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(class_report)

print("Metrics and visualizations saved!")