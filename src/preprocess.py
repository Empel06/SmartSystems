# src/preprocess.py
import os, glob
import numpy as np
import librosa
from tqdm import tqdm

SAMPLE_RATE = 44100
N_MELS = 40
HOP_LENGTH = 160
N_FFT = 512
DURATION = 2.0
SAMPLES = int(SAMPLE_RATE * DURATION)  # 88200

dataset_dir = "dataset"
out_dir = os.path.join(dataset_dir, "preprocessed")
os.makedirs(out_dir, exist_ok=True)

labels = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d != "preprocessed"]

print(f"Verwerken met target length: {SAMPLES} samples ({DURATION}s)")

for label in labels:
    files = glob.glob(os.path.join(dataset_dir, label, "*.wav"))
    arrs = []
    
    if not files:
        continue

    # Eerst bepalen wat de verwachte output shape is
    # We doen 1 dummy run om de dimensies te zien
    dummy_y = np.zeros(SAMPLES)
    dummy_mel = librosa.feature.melspectrogram(y=dummy_y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    EXPECTED_FRAMES = dummy_mel.shape[1]
    print(f"Verwachte shape per file: ({N_MELS}, {EXPECTED_FRAMES})")

    for f in tqdm(files, desc=label):
        try:
            # 1. Laad audio
            y, sr = librosa.load(f, sr=SAMPLE_RATE)
            
            # 2. HARDE FIX: Forceer lengte naar exact SAMPLES (88200)
            if len(y) < SAMPLES:
                # Te kort? Vul aan met nullen
                y = np.pad(y, (0, SAMPLES - len(y)), mode='constant')
            elif len(y) > SAMPLES:
                # Te lang? Knip af
                y = y[:SAMPLES]
            
            # 3. Nu zijn we zeker dat len(y) == 88200
            mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            log_mel = librosa.power_to_db(mel)
            
            # 4. Dubbelcheck shape (soms rondt librosa nèt anders af)
            if log_mel.shape[1] != EXPECTED_FRAMES:
                # Forceer frame count
                if log_mel.shape[1] < EXPECTED_FRAMES:
                    log_mel = np.pad(log_mel, ((0,0), (0, EXPECTED_FRAMES - log_mel.shape[1])))
                else:
                    log_mel = log_mel[:, :EXPECTED_FRAMES]
            
            arrs.append(log_mel.astype(np.float32))
            
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if arrs:
        # Nu zou stacken altijd moeten werken
        try:
            final_array = np.stack(arrs)
            np.save(os.path.join(out_dir, f"{label}.npy"), final_array)
            print(f"Saved {label}: {final_array.shape}")
        except ValueError as e:
            print(f"CRITISCHE FOUT bij stacken {label}: {e}")
            # Debug: print shapes
            shapes = [a.shape for a in arrs]
            from collections import Counter
            print(f"Gevonden shapes: {Counter(shapes)}")

print("Preprocessing klaar.")
