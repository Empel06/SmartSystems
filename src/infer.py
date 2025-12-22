# src/infer.py
import os, time, subprocess, tempfile
import torch
import numpy as np
import librosa
import soundfile as sf
from scipy.special import softmax

MODEL_PATH = "models/kws_cnn.pt"
SAMPLE_RATE = 44100       # 44100 of 48000 (werkt met jouw USB mic)
DURATION = 2.0
SAMPLES = int(SAMPLE_RATE * DURATION)

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(MODEL_PATH, map_location=device)
from train import SimpleCNN
labels = ckpt["labels"]
model = SimpleCNN(in_ch=1, num_classes=len(labels)).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("Model loaded. Labels:", labels)

def extract_log_mel(y, sr=SAMPLE_RATE, n_mels=40, hop_length=160, n_fft=512):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel)
    return log_mel

def record_with_arecord(path):
    # USB‑mic = card 3, device 0 (zie jouw `arecord -l`)
    cmd = [
        "arecord",
        "-D", "hw:3,0",
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-d", str(int(DURATION)),
        "-c", "1",
        path,
    ]
    print("Recording via arecord...")
    subprocess.run(cmd, check=True)

print("Listening... press CTRL+C to stop")
try:
    while True:
        print("Say command now...")

        # 1) Opnemen naar tijdelijke WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        record_with_arecord(wav_path)

        # 2) WAV inlezen
        y, sr = sf.read(wav_path)
        os.remove(wav_path)

        if y.ndim > 1:
            y = np.mean(y, axis=1)  # stereo -> mono

        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

        if np.max(np.abs(y)) < 0.01:
            print("Silence... try again")
            continue

        # 3) Features + inference (precies als in jouw oude code)
        feats = extract_log_mel(y)
        x = torch.tensor(feats).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out = model(x)
            probs = softmax(out.cpu().numpy()[0])
            idx = int(np.argmax(probs))
            score = float(probs[idx])
            label = labels[idx]
            print(f"Recognized: {label} ({score:.2f})")

except KeyboardInterrupt:
    print("Stopped.")
