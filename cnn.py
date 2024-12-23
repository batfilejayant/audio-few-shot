import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from PIL import Image

# Check for GPU availability
print("Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs Available: {[gpu.name for gpu in gpus]}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Using GPU for training.")
else:
    print("No GPU available. Using CPU.")

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
    for file in os.listdir(directory):
        if file.endswith(".png"):  # Ensure it's a spectrogram image file
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

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train,
    x_col="file_path",
    y_col="label",
    target_size=(224, 224),
    class_mode="binary",
    batch_size=32
)

val_generator = val_datagen.flow_from_dataframe(
    val,
    x_col="file_path",
    y_col="label",
    target_size=(224, 224),
    class_mode="binary",
    batch_size=32
)

test_generator = test_datagen.flow_from_dataframe(
    test,
    x_col="file_path",
    y_col="label",
    target_size=(224, 224),
    class_mode="binary",
    batch_size=32,
    shuffle=False
)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Save the model summary
with open(os.path.join(results_dir, "model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print("Model summary saved to model_summary.txt")

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Predictions and labels
y_true = test["label"].map({"normal": 0, "emergency": 1}).values
y_pred_prob = model.predict(test_generator).flatten()
y_pred = (y_pred_prob > 0.5).astype("int32")


# Generate classification report
class_report = classification_report(y_true, y_pred, target_names=["normal", "emergency"])
print(class_report)

# Save classification report
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(class_report)

print("Metrics and visualizations saved!")
