from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset loader for spectrograms.
        Args:
            root_dir (str): Path to the root directory (e.g., data/spectrograms).
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        
        # Discover all subdirectories (classes) under emergency/ and normal/
        for category in ["emergency", "normal"]:
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue

            for class_name in os.listdir(category_path):
                class_path = os.path.join(category_path, class_name)
                if not os.path.isdir(class_path):
                    continue

                # Add all images in the subdirectory to the dataset
                self.classes.append(f"{category}/{class_name}")
                for file_name in os.listdir(class_path):
                    if file_name.endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(class_path, file_name), len(self.classes) - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")  # Convert spectrograms to RGB
        if self.transform:
            image = self.transform(image)
        return image, label


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])
