import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import os


def hubert_training(
    model,
    train_loader,
    test_loader,
    optimizer,
    num_classes,
    feature_extractor,
    max_epochs=50,
    patience=10,
    save_path="models/hubert/hubert_model_best.pth",
    plots_path="plots/hubert",
):
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    best_accuracy = 0.0
    no_improve_epochs = 0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(max_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"[HuBERT] Epoch {epoch + 1}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Feature extraction for HuBERT input
            inputs = feature_extractor(inputs.squeeze(1), return_tensors="pt", sampling_rate=16000).input_values.to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation logic here
        model.eval()
        val_loss, correct, total = 0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = feature_extractor(inputs.squeeze(1), return_tensors="pt", sampling_rate=16000).input_values.to(device)

                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        val_accuracy = 100 * correct / total
        val_losses.append(val_loss / len(test_loader))
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}: Val Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            no_improve_epochs = 0
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping.")
            break
