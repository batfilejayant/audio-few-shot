import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import json

# Enable GPU and multi-core processing
def set_device_and_cores():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count())
    print(f"Using device: {device}")
    print(f"Using {os.cpu_count()} CPU cores")
    return device

device = set_device_and_cores()

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing the spectrogram images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Load data from emergency and normal directories
        main_dirs = ['emergency', 'normal']
        for main_dir in main_dirs:
            main_path = os.path.join(root_dir, main_dir)
            if os.path.isdir(main_path):
                for sub_dir in os.listdir(main_path):
                    sub_path = os.path.join(main_path, sub_dir)
                    if os.path.isdir(sub_path):
                        for img_file in os.listdir(sub_path):
                            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                                self.data.append((os.path.join(sub_path, img_file), main_dir + '_' + sub_dir))

        # Map class names to indices
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(set([d[1] for d in self.data])))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label

# Function to create data loaders for MAML tasks
def create_task_dataloaders(root_dir, num_tasks, num_samples_per_task, transform=None):
    """
    Args:
        root_dir (str): Root directory containing the spectrogram images.
        num_tasks (int): Number of tasks to create.
        num_samples_per_task (int): Number of samples per task (split into train and val).
        transform (callable, optional): Transformations for data augmentation.

    Returns:
        list: List of task-specific dataloaders [(train_loader, val_loader), ...].
    """
    dataset = SpectrogramDataset(root_dir, transform=transform)
    tasks = []

    for _ in range(num_tasks):
        # Select two random classes
        selected_classes = random.sample(list(dataset.class_to_idx.keys()), k=2)
        selected_data = [(img, label) for img, label in dataset.data if label in selected_classes]

        # Create a subset dataset
        class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        selected_dataset = [(img, class_to_idx[label]) for img, label in selected_data]

        # Shuffle and split into train and validation
        random.shuffle(selected_dataset)
        train_size = num_samples_per_task // 2
        val_size = num_samples_per_task - train_size

        train_data = selected_dataset[:train_size]
        val_data = selected_dataset[train_size:train_size + val_size]

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=os.cpu_count())
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=os.cpu_count())

        tasks.append((train_loader, val_loader))

    return tasks

# Train model with metrics tracking and early stopping
def train_model_with_metrics(model, dataloaders, criterion, optimizer, num_epochs=100, patience=10, model_save_dir="models/maml", plot_dir="plots/maml"):
    metrics = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': []
    }

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    best_loss = float('inf')
    patience_counter = 0
    all_labels = []
    all_preds = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_labels = []
        epoch_preds = []

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())

        all_labels.extend(epoch_labels)
        all_preds.extend(epoch_preds)
        epoch_loss /= len(dataloaders['train'].dataset)
        accuracy = accuracy_score(epoch_labels, epoch_preds)
        precision = precision_score(epoch_labels, epoch_preds, average='weighted')
        recall = recall_score(epoch_labels, epoch_preds, average='weighted')
        f1 = f1_score(epoch_labels, epoch_preds, average='weighted')
        roc_auc = roc_auc_score(epoch_labels, epoch_preds, average='weighted', multi_class='ovo')

        metrics['loss'].append(epoch_loss)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

        # Save the model if loss improves
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"best_model_epoch_{epoch + 1}.pth"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Generate classification report
    report = classification_report(all_labels, all_preds, output_dict=True)
    report_path = os.path.join(plot_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Classification report saved to {report_path}")

    return metrics, report

# Generate plots for metrics
def plot_metrics(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for metric, values in metrics.items():
        plt.figure()
        plt.plot(values, label=metric)
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{metric}.png'))
        plt.close()

# Save model summary
def save_model_summary(model, input_size, plot_dir):
    from torchsummary import summary
    os.makedirs(plot_dir, exist_ok=True)
    summary_str = []
    def print_fn(line):
        summary_str.append(line)
    summary(model, input_size=input_size, print_fn=print_fn, device=device)
    summary_path = os.path.join(plot_dir, "model_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_str))
    print(f"Model summary saved to {summary_path}")

# Example usage
if __name__ == "__main__":
    root_dir = "data/spectrograms"
    num_tasks = 10
    num_samples_per_task = 20

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    task_dataloaders = create_task_dataloaders(root_dir, num_tasks, num_samples_per_task, transform=transform)

    # Replace with your model, criterion, and optimizer
    # Example placeholders:
    # model = YourModel().to(device)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Replace with actual train/val split
    # dataloaders = {'train': train_loader, 'val': val_loader}

    # Example training
    # metrics, report = train_model_with_metrics(model, dataloaders, criterion, optimizer)

    # plot_metrics(metrics, "plots/maml")
    # save_model_summary(model, (3, 128, 128), "plots/maml")
