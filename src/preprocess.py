# src/preprocess.py
import os, glob
import numpy as np
import librosa
from tqdm import tqdm

SAMPLE_RATE = 44100
N_MELS = 40
HOP_LENGTH = 160
N_FFT = 512
DURATION = 2.0  # seconds
SAMPLES = int(SAMPLE_RATE * DURATION)

dataset_dir = "dataset"
out_dir = os.path.join(dataset_dir, "preprocessed")
os.makedirs(out_dir, exist_ok=True)

labels = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d != "preprocessed"]

print("Labels:", labels)

for label in labels:
    files = glob.glob(os.path.join(dataset_dir, label, "*.wav"))
    arrs = []
    for f in tqdm(files, desc=label):
        y, sr = librosa.load(f, sr=SAMPLE_RATE)
        # pad / trim
        if len(y) < SAMPLES:
            y = np.pad(y, (0, SAMPLES - len(y)))
        else:
            y = y[:SAMPLES]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel)
        arrs.append(log_mel.astype(np.float32))
    if arrs:
        np.save(os.path.join(out_dir, f"{label}.npy"), np.stack(arrs))
        print(f"Saved {label}: {len(arrs)} examples")
print("Preprocessing finished.")
