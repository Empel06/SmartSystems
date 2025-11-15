# src/record.py
import argparse, os
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 2.0  # seconds

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="label folder name")
parser.add_argument("--n", type=int, default=30, help="number of samples")
parser.add_argument("--duration", type=float, default=DURATION, help="duration per sample (s)")
args = parser.parse_args()

out_dir = os.path.join("dataset", args.label)
os.makedirs(out_dir, exist_ok=True)

print(f"Recording {args.n} samples for label '{args.label}' into {out_dir}")
for i in range(args.n):
    input(f"Press ENTER and say the phrase ({i+1}/{args.n})...")
    print("Recording...")
    data = sd.rec(int(args.duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    fname = os.path.join(out_dir, f"{args.label}_{i:03d}.wav")
    sf.write(fname, data, SAMPLE_RATE)
    print(f"Saved: {fname}")
print("Done.")
