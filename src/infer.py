# src/infer.py
import os, time
import torch
import sounddevice as sd
import numpy as np
import librosa
from scipy.special import softmax

MODEL_PATH = "models/kws_cnn.pt"
SAMPLE_RATE = 16000
DURATION = 2.0
SAMPLES = int(SAMPLE_RATE * DURATION)

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(MODEL_PATH, map_location=device)
from train import SimpleCNN  # ensure train.py in same folder, or copy class to this file
labels = ckpt["labels"]
model = SimpleCNN(in_ch=1, num_classes=len(labels)).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("Model loaded. Labels:", labels)

def extract_log_mel(y, sr=SAMPLE_RATE, n_mels=40, hop_length=160, n_fft=512):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel)
    return log_mel

print("Listening... press CTRL+C to stop")
try:
    while True:
        print("Say command now...")
        data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        y = data.flatten()
        if np.max(np.abs(y)) < 0.01:
            print("Silence... try again")
            continue
        feats = extract_log_mel(y)
        x = torch.tensor(feats).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,n_mels,frames)
        with torch.no_grad():
            out = model(x)
            probs = softmax(out.cpu().numpy()[0])
            idx = int(np.argmax(probs))
            score = float(probs[idx])
            label = labels[idx]
            print(f"Recognized: {label} ({score:.2f})")
except KeyboardInterrupt:
    print("Stopped.")
