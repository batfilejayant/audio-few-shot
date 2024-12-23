import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config, Wav2Vec2FeatureExtractor
from torch.optim import Adam
from PIL import Image
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from tqdm import tqdm
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Custom Dataset for Spectrogram Images
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.classes = []
        self.transform = transform

        # Traverse the emergency/normal directories
        for category in ["emergency", "normal"]:
            category_path = os.path.join(root_dir, category)
            for class_idx, class_name in enumerate(sorted(os.listdir(category_path))):
                class_path = os.path.join(category_path, class_name)
                if os.path.isdir(class_path):
                    for file_name in os.listdir(class_path):
                        if file_name.endswith((".png", ".jpg", ".jpeg")):
                            file_path = os.path.join(class_path, file_name)
                            self.samples.append((file_path, class_idx))
                    self.classes.append(f"{category}/{class_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# Feature Extractor Wrapper
class FlattenedFeatureExtractor:
    def __init__(self, feature_extractor, device):
        self.feature_extractor = feature_extractor
        self.device = device

    def __call__(self, inputs):
        inputs = inputs.squeeze(1)  # Remove channel dimension: [B, 1, H, W] -> [B, H, W]
        inputs = inputs.view(inputs.size(0), -1)  # Flatten: [B, H, W] -> [B, time_steps]
        inputs = self.feature_extractor(inputs, return_tensors="pt", sampling_rate=16000).input_values
        return inputs.to(self.device)


# Training Function (simplified version since full training loop isn't provided)
def hubert_training(model, train_loader, test_loader, optimizer, num_classes, feature_extractor, max_epochs, patience, save_path, plots_path):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                inputs = feature_extractor(inputs)
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs.logits, targets.to(device))

                # Check for NaN
                if torch.isnan(loss):
                    logging.warning(f"NaN loss detected at epoch {epoch}, batch {batch_idx}. Skipping batch.")
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        # Validation loop (simplified)
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in test_loader:
                inputs = feature_extractor(inputs)
                outputs = model(inputs)
                val_loss += torch.nn.functional.cross_entropy(outputs.logits, targets.to(device)).item()

            avg_val_loss = val_loss / len(test_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break


def main():
    # Paths
    root_dir = "data/spectrograms"
    save_path = "models/hubert/hubert_model_best.pth"
    plots_path = "plots/hubert"

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Image Transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize spectrograms
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
    ])

    # Load Dataset
    logging.info("Loading dataset...")
    dataset = SpectrogramDataset(root_dir=root_dir, transform=transform)

    # Train-Test Split
    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    logging.info(f"Total samples: {len(dataset)}")
    logging.info(f"Training samples: {train_size}, Testing samples: {test_size}")

    # Data Loaders (with CPU parallelism)
    batch_size = 32
    num_workers = os.cpu_count()  # Use all CPU cores for data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Initialize HuBERT Model
    logging.info("\nInitializing HuBERT Model...")
    num_classes = len(dataset.classes)
    config = Wav2Vec2Config.from_pretrained("facebook/hubert-base-ls960", num_labels=num_classes)
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/hubert-base-ls960", config=config)
    
    # Adapt the model for your use
    model.classifier = torch.nn.Linear(config.hidden_size, num_classes)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Feature Extractor for HuBERT
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    flattened_feature_extractor = FlattenedFeatureExtractor(feature_extractor, device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=2e-5)

    # Train the Model
    hubert_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        num_classes=num_classes,
        feature_extractor=flattened_feature_extractor,
        max_epochs=100,
        patience=50,
        save_path=save_path,
        plots_path=plots_path,
    )

if __name__ == "__main__":
    # Multiprocessing for DataLoader compatibility
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()