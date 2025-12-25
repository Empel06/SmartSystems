# dataset/ - Training Data Management

## Location

SmartSystems/dataset/ contains all training data for the model.

## Description

This folder contains:
- Raw audio samples - Original recordings per command
- Preprocessed features - Extracted mel-spectrograms
- Data organization - Structured by command labels
- Metadata - Information about each sample

## Structure

```
dataset/
├── pause_timer/             # Audio samples: pause timer
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ... (x samples)
├── preprocessed/            # Processed features
│   ├── pause_timer.npy
│   ├── room_temp.npy
│   ├── silence.npy
│   ├── start_timer.npy
│   └── work_timer.npy
├── room_temp/               # Audio samples: room temperature
│   ├── sample_001.wav
│   └── ... (x samples)
├── silence/                 # Audio samples: silence
│   ├── sample_001.wav
│   └── ... (x samples)
├── start_timer/             # Audio samples: start timer
│   ├── sample_001.wav
│   └── ... (x samples)
└── work_timer/              # Audio samples: work timer
    ├── sample_001.wav
    └── ... (x samples)
```

## Data Labels

| Label | Usage | Samples | Purpose |
|-------|-------|---------|---------|
| start_timer | Start productivity timer | x| Begin work session |
| pause_timer | Pause current timer | x | Temporarily stop |
| room_temp | Check room temperature | x | Environmental info |
| pomodoro | Start Pomodoro session | x | 25/5 focus technique |
| silence | Ignore/no command | x | Background noise |
| Total | | x | Training dataset |

## Recording Training Data

### Record New Samples

```bash
# Record 30 samples for "start_timer" command
python src/record.py --label start_timer --n 30

# Record with custom settings
python src/record.py \
  --label start_timer \
  --n 30 \
  --duration 3 \
  --sr 16000 \
  --device 1

# Record silence samples
python src/record.py --label silence --n 20
```

### Recording Guidelines

Do:
- Speak clearly and naturally
- Vary your voice (different people, pitches)
- Record in different acoustic environments
- Use good microphone placement
- Speak at normal conversation volume

Don't:
- Whisper or shout
- Have background noise
- Speak too fast or slow
- Repeat exactly the same each time
- Record with poor audio quality

### Recording Parameters

| Parameter | Default | Recommended |
|-----------|---------|-------------|
| Duration | 2 sec | 2-4 seconds |
| Sample Rate | 441000 Hz | 441000 Hz |
| Channels | 1 (mono) | 1 (mono) |
| Bit Depth | 16-bit | 16-bit |

## Data Preprocessing

### Preprocess All Data

```bash
# Generate features from raw audio
python src/preprocess.py

# Preprocess specific label only
python src/preprocess.py --label start_timer

# Custom parameters
python src/preprocess.py \
  --n_mels 128 \
  --n_fft 512 \
  --hop_length 160
```

### Output Format

After preprocessing, features are saved as .npy files:

```
dataset/preprocessed/
├── start_timer.npy          (30, 128, 101)
├── pause_timer.npy          (30, 128, 101)
├── room_temp.npy            (30, 128, 101)
├── pomodoro.npy             (30, 128, 101)
└── silence.npy              (20, 128, 101)
```

Shape Explanation:
- 30 = Number of samples
- 128 = Mel frequency bins
- 101 = Time frames (approximately 3 seconds at 16kHz)

### Feature Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| n_mels | 128 | Mel-frequency bins |
| n_fft | 512 | FFT window size |
| hop_length | 160 | Samples per hop |
| sr | 16000 | Sample rate (Hz) |
| norm | slaney | Normalization |

## Data Augmentation

### Augmentation Techniques

```python
from src.augment import AudioAugmenter

augmenter = AudioAugmenter()

# Apply random augmentations
augmented = augmenter.augment(audio, sr)
```

Available Techniques:

1. Time Stretching - Speed up/slow down
   ```python
   audio_stretched = librosa.effects.time_stretch(audio, rate=1.1)
   # 1.0 = normal, 1.1 = 10% faster, 0.9 = 10% slower
   ```

2. Pitch Shifting - Change pitch without speed
   ```python
   audio_pitched = librosa.effects.pitch_shift(audio, sr=16000, n_steps=2)
   # Shifts by semitones
   ```

3. Adding Noise - Noise robustness
   ```python
   noise = np.random.normal(0, 0.005, len(audio))
   audio_noisy = audio + noise
   ```

4. Volume Adjustment - Scale amplitude
   ```python
   audio_loud = audio * 1.2   # 20% louder
   audio_quiet = audio * 0.8  # 20% quieter
   ```

5. Time Shifting - Random time offset
   ```python
   shift = np.random.randint(-1600, 1600)  # +/- 0.1 sec
   audio_shifted = np.roll(audio, shift)
   ```

### Apply Augmentation to Dataset

```bash
# Augment dataset (double the samples)
python src/augment.py --input dataset/ --output dataset/augmented

# Apply specific augmentations
python src/augment.py --techniques time_stretch pitch_shift --ratio 2
```

## Data Statistics

### Dataset Overview

```
Total Samples: 150
Total Duration: 450 seconds (7.5 minutes)
Sample Rate: 16000 Hz
Channels: 1 (Mono)
Bit Depth: 16-bit

Distribution:
├── start_timer:  30 samples (33%)
├── pause_timer:  30 samples (33%)
├── room_temp:    30 samples (33%)
├── pomodoro:     30 samples (33%)
└── silence:      20 samples (13%)
```

### To Improve Accuracy (95% to 99%+)

1. Add More Samples
   ```
   Target: 100-200 samples per label
   Current: 30 samples per label
   Impact: +5-10% accuracy
   ```

2. Diverse Speakers
   ```
   Current: Limited speakers
   Target: 10+ different people
   Impact: Better generalization
   ```

3. Different Environments
   ```
   Locations to record in:
   - Quiet office
   - Noisy café
   - Car interior
   - Outdoor/street
   Impact: Robustness to noise
   ```

4. Data Augmentation
   ```
   Apply automatically:
   - Time stretching (0.8-1.2x)
   - Pitch shifting (±2 semitones)
   - Background noise
   Impact: +3-5% accuracy
   ```

5. Acoustic Variations
   ```
   Vary recording conditions:
   - Different microphones
   - Distance from mic
   - Recording levels
   Impact: Hardware independence
   ```