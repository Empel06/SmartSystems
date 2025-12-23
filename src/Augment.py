# src/augment.py
import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path


SAMPLE_RATE = 44100  # Same as in train.py and infer.py


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise


def change_pitch(data, sr, pitch_steps=2):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_steps)


def change_speed(data, speed_factor=1.1):
    return librosa.effects.time_stretch(data, rate=speed_factor)


def augment_dataset(folder_path):
    """Generate 4 augmented versions of each original audio file"""
    if not os.path.exists(folder_path):
        print(f"Error: folder {folder_path} does not exist")
        return
    
    print(f"Starting augmentation of: {folder_path}")
    
    # Find all original .wav files (skip already augmented files)
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav') and "_aug_" not in f]
    
    if not files:
        print("No audio files found to augment")
        return
    
    count = 0
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            data, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            filename_no_ext = os.path.splitext(file)[0]
            
            # 1. Noise variant
            data_noise = add_noise(data)
            sf.write(os.path.join(folder_path, f"{filename_no_ext}_aug_noise.wav"), data_noise, sr)
            count += 1
            
            # 2. Pitch variant (lower)
            data_pitch_low = change_pitch(data, sr, -2)
            sf.write(os.path.join(folder_path, f"{filename_no_ext}_aug_pitch_low.wav"), data_pitch_low, sr)
            count += 1
            
            # 3. Pitch variant (higher)
            data_pitch_high = change_pitch(data, sr, 2)
            sf.write(os.path.join(folder_path, f"{filename_no_ext}_aug_pitch_high.wav"), data_pitch_high, sr)
            count += 1
            
            # 4. Speed variant (faster)
            # Fix length to match original for consistent dimensions
            data_speed = change_speed(data, 1.1)
            if len(data_speed) < len(data):
                data_speed = np.pad(data_speed, (0, len(data) - len(data_speed)))
            else:
                data_speed = data_speed[:len(data)]
            sf.write(os.path.join(folder_path, f"{filename_no_ext}_aug_speed.wav"), data_speed, sr)
            count += 1
            
            print(f"Augmented: {file} -> 4 variants")
            
        except Exception as e:
            print(f"Error augmenting {file}: {e}")
            continue
    
    print(f"Done! Generated {count} augmented files from {len(files)} originals.")
    print(f"Your dataset is now ~5x larger (original + 4 variants per file)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment audio dataset')
    parser.add_argument('--label', default='', help='Specific label folder to augment (e.g., start_timer)')
    parser.add_argument('--all', action='store_true', help='Augment all labels in dataset/')
    
    args = parser.parse_args()
    
    if args.all:
        dataset_root = "dataset"
        if os.path.exists(dataset_root):
            for label in os.listdir(dataset_root):
                label_path = os.path.join(dataset_root, label)
                if os.path.isdir(label_path):
                    augment_dataset(label_path)
    elif args.label:
        augment_dataset(f"dataset/{args.label}")
    else:
        # Default: augment all labels
        dataset_root = "dataset"
        if os.path.exists(dataset_root):
            for label in os.listdir(dataset_root):
                label_path = os.path.join(dataset_root, label)
                if os.path.isdir(label_path):
                    augment_dataset(label_path)