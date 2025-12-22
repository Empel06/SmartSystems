import argparse, os, subprocess, tempfile
import soundfile as sf
import numpy as np


SAMPLE_RATE = 44100  # idem als in infer.py
DURATION = 2.0       # seconds


def get_next_file_number(label_dir):
    """
    Vind het volgende beschikbare nummertje
    Zoekt naar bestaande .wav files en geeft het volgende nummer
    """
    if not os.path.exists(label_dir):
        return 0
    
    # Zoek alle WAV files
    wav_files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
    
    if not wav_files:
        return 0
    
    # Extract nummers uit filenames (bijv: "start_timer_025.wav" -> 25)
    numbers = []
    for filename in wav_files:
        try:
            number_str = filename.split('_')[-1].replace('.wav', '')
            numbers.append(int(number_str))
        except (ValueError, IndexError):
            continue
    
    if numbers:
        return max(numbers) + 1
    return 0


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


parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="label folder name")
parser.add_argument("--n", type=int, default=30, help="number of samples")
parser.add_argument("--duration", type=float, default=DURATION, help="duration per sample (s)")
parser.add_argument("--list", action="store_true", help="Toon bestaande samples")
args = parser.parse_args()


out_dir = os.path.join("dataset", args.label)
os.makedirs(out_dir, exist_ok=True)


def list_existing_samples(output_dir="dataset"):
    """Toon hoeveel samples je al hebt per label"""
    if not os.path.exists(output_dir):
        print("Dataset directory bestaat niet")
        return
    
    print("\nHuidige dataset:")
    print("-" * 50)
    
    for label in sorted(os.listdir(output_dir)):
        label_path = os.path.join(output_dir, label)
        if os.path.isdir(label_path):
            wav_files = [f for f in os.listdir(label_path) if f.endswith('.wav')]
            if wav_files:
                numbers = []
                for f in wav_files:
                    try:
                        num = int(f.split('_')[-1].replace('.wav', ''))
                        numbers.append(num)
                    except:
                        pass
                
                if numbers:
                    print(f"  {label}: {len(wav_files)} samples")
                    print(f"     Nummers: {min(numbers):03d} tot {max(numbers):03d}")
    print("-" * 50 + "\n")


if args.list:
    list_existing_samples()
else:
    # Vind volgende nummer
    start_number = get_next_file_number(out_dir)
    
    print(f"\nLabel: {args.label}")
    print(f"Start nummer: {start_number:03d}")
    print(f"Eindig nummer: {start_number + args.n - 1:03d}")
    print(f"Duration per sample: {args.duration}s")
    print(f"Locatie: {out_dir}\n")
    
    for i in range(args.n):
        # BELANGRIJK: gebruik start_number + i in plaats van alleen i
        current_number = start_number + i
        
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
        
        # VERANDERD: gebruik current_number in plaats van i
        fname = os.path.join(out_dir, f"{args.label}_{current_number:03d}.wav")
        sf.write(fname, y, SAMPLE_RATE)
        print(f"Saved: {fname}")
    
    print(f"\nKlaar! {args.n} samples opgeslagen.")
    print(f"   Nummers: {start_number:03d} tot {start_number + args.n - 1:03d}\n")