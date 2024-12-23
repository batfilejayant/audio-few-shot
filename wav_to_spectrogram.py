import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Update matplotlib settings to avoid excessive warnings
mpl.rcParams['figure.max_open_warning'] = 0  # Disable max open warning

def wav_to_spectrogram(wav_file, output_image):
    """
    Converts a WAV file to a spectrogram and saves it as an image.

    Args:
        wav_file (str): Path to the input WAV file.
        output_image (str): Path to save the spectrogram image.
    """
    # Load the audio file
    y, sr = librosa.load(wav_file, sr=None)  # y is the audio time series, sr is the sampling rate

    # Generate the Short-Time Fourier Transform (STFT) of the audio
    S = librosa.stft(y)

    # Convert the amplitude to decibels
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')  # Remove axes for a clean image

    # Display the spectrogram
    img = librosa.display.specshow(S_db, sr=sr, ax=ax, cmap='viridis')

    # Save the spectrogram
    fig.savefig(output_image, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Explicitly close the figure


def process_file(args):
    """
    Process a single file for spectrogram generation.

    Args:
        args (tuple): Contains (wav_file, output_image).
    """
    wav_file, output_image = args

    # Skip if the spectrogram already exists
    if os.path.exists(output_image):
        return

    wav_to_spectrogram(wav_file, output_image)
    print(f"Converted: {wav_file} -> {output_image}")

def process_directory_gpu(main_folder):
    """
    Walks through the directory structure, collects all WAV files for processing.

    Args:
        main_folder (str): Path to the main folder containing subdirectories and WAV files.

    Returns:
        list: List of tuples containing (wav_file, output_image).
    """
    tasks = []
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.wav'):
                wav_file = os.path.join(root, file)
                # Create corresponding spectrogram directory
                relative_path = os.path.relpath(root, main_folder)
                spectrogram_dir = os.path.join(main_folder, 'spectrograms', relative_path)
                os.makedirs(spectrogram_dir, exist_ok=True)
                # Output image path
                output_image = os.path.join(spectrogram_dir, f"{os.path.splitext(file)[0]}.png")
                tasks.append((wav_file, output_image))
    return tasks

def main():
    """
    Main function to process a single main folder using GPU and multiple threads.
    """
    main_folder = '../normal-sounds'  # Replace with your folder path

    # Collect tasks from the main folder
    tasks = process_directory_gpu(main_folder)

    # Use ThreadPoolExecutor for parallel file processing with a progress bar
    num_cores = os.cpu_count()  # Get the number of available CPU cores
    print(f"Using {num_cores} threads for processing.")
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        list(tqdm(executor.map(process_file, tasks), total=len(tasks), desc="Processing files"))

if __name__ == "__main__":
    main()