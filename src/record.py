# src/record.py
import argparse, os, subprocess, tempfile
import soundfile as sf
import numpy as np

SAMPLE_RATE = 44100  # idem als in infer.py
DURATION = 2.0       # seconds

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="label folder name")
parser.add_argument("--n", type=int, default=30, help="number of samples")
parser.add_argument("--duration", type=float, default=DURATION, help="duration per sample (s)")
args = parser.parse_args()

out_dir = os.path.join("dataset", args.label)
os.makedirs(out_dir, exist_ok=True)

def record_with_arecord(path, duration):
    cmd = [
        "arecord",
        "-D", "hw:3,0",      # USB-mic
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-d", str(int(duration)),
        "-c", "1",
        path,
    ]
    print("Recording via arecord...")
    subprocess.run(cmd, check=True)

print(f"Recording {args.n} samples for label '{args.label}' into {out_dir}")
for i in range(args.n):
    input(f"Press ENTER and say the phrase ({i+1}/{args.n})...")
    print("Recording...")

    # 1) neem tijdelijk op met arecord
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    record_with_arecord(tmp_path, args.duration)

    # 2) lees binnen en schrijf naar dataset
    y, sr = sf.read(tmp_path)
    os.remove(tmp_path)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != SAMPLE_RATE:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    fname = os.path.join(out_dir, f"{args.label}_{i:03d}.wav")
    sf.write(fname, y, SAMPLE_RATE)
    print(f"Saved: {fname}")

print("Done.")
