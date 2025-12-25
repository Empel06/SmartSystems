# src/ - Core Application Code

## Location

SmartSystems/src/ contains all core application code for the SmartSystems project.

## Description

This is the heart of the project. All modules here work together to:
- Read audio from the microphone
- Recognize commands
- Execute actions
- Provide feedback via TTS

## Structure

```
src/
├── __pycache__/              # Python cache (generated)
├── app.py                    # Main application entry point
├── record.py                 # Audio recording utility
├── train.py                  # Model training script
├── infer.py                  # Real-time inference and prediction
├── preprocess.py             # Audio preprocessing
├── augment.py                # Data augmentation
├── commands.py               # Command definitions
├── config.py                 # Configuration management
├── utils.py                  # Utility functions
├── smart_assistant.py        # Core AI logic
└── requirements.txt          # Dependencies
```

## Key Modules

### 1. app.py - MAIN APPLICATION

The main entry point of the application.

Responsibilities:
- Initialize all modules
- Setup audio input
- Manage command recognition loop
- Handle user interactions
- Shutdown gracefully

Run:
```bash
python src/app.py
```

Options:
```bash
python src/app.py --debug           # Verbose logging
python src/app.py --model models/final_model.onnx  # Custom model
python src/app.py --device cpu      # Force CPU (or cuda)
```

### 2. record.py - AUDIO RECORDING

Records audio samples from users for training.

Usage:
```bash
# Record 30 samples for "start_timer" command
python src/record.py --label start_timer --n 30

# Record with custom parameters
python src/record.py --label pause_timer --n 20 --duration 3 --sr 16000

# Available labels:
# - start_timer
# - pause_timer
# - room_temp
# - pomodoro
# - silence
```

Output:
```
dataset/start_timer/
├── sample_001.wav
├── sample_002.wav
├── sample_003.wav
└── ...
```

Parameters:
- --label: Command label
- --n: Number of samples to record
- --duration: Recording duration (default: 3 seconds)
- --sr: Sample rate (default: 16000 Hz)
- --device: Audio device index

### 3. preprocess.py - PREPROCESSING

Converts raw audio to features for model training.

Usage:
```bash
# Preprocess all recorded audio
python src/preprocess.py

# Preprocess specific label
python src/preprocess.py --label start_timer

# Custom parameters
python src/preprocess.py --n_mels 128 --n_fft 512
```

What it does:
1. Load all WAV files from dataset/
2. Convert to log-mel spectrogram
3. Normalize features
4. Save as .npy files in dataset/preprocessed/

Output:
```
dataset/preprocessed/
├── start_timer.npy
├── pause_timer.npy
├── room_temp.npy
├── pomodoro.npy
└── silence.npy
```

Hyperparameters:
- n_mels: Mel-frequency bins (default: 128)
- n_fft: FFT window size (default: 512)
- hop_length: Hop size (default: 160)
- sr: Sample rate (default: 16000)

### 4. train.py - MODEL TRAINING

Trains the CNN model for command recognition.

Usage:
```bash
# Train model from preprocessed data
python src/train.py

# With custom parameters
python src/train.py --epochs 100 --batch_size 32 --lr 0.001

# Resume from checkpoint
python src/train.py --resume models/checkpoint.pt
```

What it does:
1. Load preprocessed features
2. Create train/validation split
3. Train CNN model
4. Evaluate on validation set
5. Save best model

Output:
```
models/
├── kws_cnn.pt              # Final model
├── best_model.pt           # Best validation score
└── checkpoint.pt           # Last checkpoint
```

Hyperparameters:
- --epochs: Training epochs (default: 50)
- --batch_size: Batch size (default: 16)
- --lr: Learning rate (default: 0.001)
- --dropout: Dropout rate (default: 0.3)
- --augment: Enable data augmentation

Expected Metrics:
- Accuracy: 92-96% (with 30 samples/label)
- Training time: 5-10 minutes (Pi 4)
- Model size: 5-10 MB

### 5. infer.py - REAL-TIME INFERENCE

Performs real-time command recognition on live audio.

Usage:
```bash
# Start real-time inference
python src/infer.py

# Use custom model
python src/infer.py --model models/final_model.onnx

# Set confidence threshold
python src/infer.py --threshold 0.7

# Verbose output
python src/infer.py --verbose
```

Output:
```
Listening for commands...
Recognized: start_timer (confidence: 0.94)
Recognized: room_temp (confidence: 0.88)
Recognized: silence (confidence: 0.72)
```

Parameters:
- --model: Model path
- --threshold: Confidence threshold (0-1)
- --chunk_size: Audio chunk size
- --sr: Sample rate

### 6. smart_assistant.py - CORE LOGIC

The core logic for command handling and responses.

Key Classes:

```python
class SmartAssistant:
    """Main assistant class"""
    
    def __init__(self, model_path, config_path):
        # Initialize model, TTS, timers
        
    def process_command(self, command, confidence):
        # Execute command action
        
    def generate_response(self, command):
        # Create natural language response
        
    def speak(self, text):
        # Text-to-speech output
```

Supported Commands:

| Command | Action |
|---------|--------|
| start_timer | Start work timer |
| pause_timer | Pause current timer |
| room_temp | Report room temperature |
| pomodoro | Start Pomodoro session |
| silence | Ignore/no action |

### 7. config.py - CONFIGURATION

Manages all configuration settings.

Usage:
```python
from src.config import Config

config = Config('config.json')
sr = config.get('audio.sample_rate')  # 16000
model_path = config.get('model.path')
```

Example config.json:
```json
{
  "audio": {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024
  },
  "model": {
    "path": "models/final_model.onnx",
    "device": "cpu",
    "threshold": 0.7
  },
  "tts": {
    "enabled": true,
    "rate": 150,
    "volume": 1.0
  }
}
```

### 8. utils.py - UTILITIES

Helper functions used by other modules.

Common Functions:
```python
# Audio utilities
load_audio(filepath)
save_audio(audio, sr, filepath)
normalize_audio(audio)

# Feature utilities
extract_mfcc(audio, sr)
extract_mel_spectrogram(audio, sr)

# File utilities
get_label_files(label)
create_dataset_dirs()

# Logger utilities
setup_logger(name, level)
log_metrics(metrics)
```

### 9. augment.py - DATA AUGMENTATION

Provides data augmentation techniques to expand training data.

Techniques:
- Time stretching
- Pitch shifting
- Adding background noise
- Volume adjustment
- Time shifting

Usage:
```python
from src.augment import AudioAugmenter

augmenter = AudioAugmenter()
augmented = augmenter.augment(audio, sr)
```

### 10. commands.py - COMMAND DEFINITIONS

Defines all supported commands.

```python
COMMANDS = {
    'start_timer': {
        'action': 'start_timer',
        'response': 'Timer started',
        'duration': 25  # minutes
    },
    'pause_timer': {
        'action': 'pause_timer',
        'response': 'Timer paused'
    },
    'room_temp': {
        'action': 'get_temperature',
        'response': 'The room temperature is'
    },
    'pomodoro': {
        'action': 'start_pomodoro',
        'response': 'Starting pomodoro session',
        'work_duration': 25,
        'break_duration': 5
    }
}
```

## Audio Processing Pipeline

```
Microphone
    ↓
record.py (Raw WAV)
    ↓
preprocess.py (Features)
    ↓
CNN Model (Prediction)
    ↓
Command Recognition
    ↓
smart_assistant.py (Action)
    ↓
TTS Output
```

## Testing

### Unit Tests
```bash
# Test individual modules
python -m pytest src/tests/ -v

# Test specific module
python -m pytest src/tests/test_preprocess.py -v

# Coverage report
python -m pytest src/tests/ --cov=src
```

### Integration Test
```bash
# Test complete pipeline
python src/test_pipeline.py

# Should output:
# OK - Audio loading
# OK - Preprocessing
# OK - Inference
# OK - Command execution
```

### Manual Testing
```bash
# 1. Record a sample
python src/record.py --label test_command --n 1

# 2. Preprocess it
python src/preprocess.py --label test_command

# 3. Train (if needed)
python src/train.py --epochs 10

# 4. Test inference
python src/infer.py
```

## Configuration

### config.py Example

```python
# Load configuration
from src.config import Config

config = Config('src/config.json')

# Access settings
audio_sr = config['audio']['sample_rate']
model_path = config['model']['path']
```

### config.json

```json
{
  "audio": {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_duration": 1.0,
    "silence_threshold": -40
  },
  
  "preprocessing": {
    "n_mels": 128,
    "n_fft": 512,
    "hop_length": 160
  },
  
  "model": {
    "path": "models/final_model.onnx",
    "device": "cpu",
    "confidence_threshold": 0.7
  },
  
  "tts": {
    "enabled": true,
    "engine": "pyttsx3",
    "rate": 150,
    "volume": 1.0
  }
}
```

## Data Flow

```
Input: Spoken command
  ↓
record.py: Convert to WAV (16kHz, mono)
  ↓
preprocess.py: Extract mel-spectrogram
  ↓
train.py: Normalize and batch
  ↓
CNN Model: Classify (softmax)
  ↓
smart_assistant.py: Interpret and respond
  ↓
config.py: Load action config
  ↓
TTS Engine: Speak response
  ↓
Output: Action executed + Audio response
```

## Troubleshooting

### "No audio device found"
```python
# List available devices
import sounddevice as sd
print(sd.query_devices())

# Specify device in record.py
python src/record.py --label test --device 1
```

### "Low accuracy on inference"
```
Solutions:
1. Record more samples (50-100 per label)
2. Use data augmentation (augment.py)
3. Train longer (increase epochs)
4. Adjust confidence threshold
```

### "High latency (greater than 1 second)"
```
Solutions:
1. Use ONNX model instead of PyTorch
2. Reduce audio chunk size
3. Enable GPU acceleration
4. Profile with: python -m cProfile src/infer.py
```

### "Memory error during training"
```
Solutions:
1. Reduce batch size: --batch_size 8
2. Reduce number of mel bins: --n_mels 64
3. Limit number of samples: --max_samples 1000
```

## Dependencies

Core dependencies used in src/:
```
torch>=2.0.0
librosa>=0.10.0
numpy>=1.24.0
scipy>=1.10.0
sounddevice>=0.4.6
soundfile>=0.12.0
```

See requirements.txt for complete list.

## Performance Tips

1. Use ONNX models - 2-3x faster inference
2. Optimize batch size - Balance between speed and accuracy
3. Reduce mel frequency bins - Lower memory usage
4. Enable GPU - If available (CUDA/ROCm)
5. Profile code - Find bottlenecks

```bash
# Profile inference speed
python -m cProfile -s cumulative src/infer.py

# Profile memory usage
python -m memory_profiler src/infer.py
```

## Development Workflow

### 1. Record New Command
```bash
python src/record.py --label new_command --n 30
```

### 2. Preprocess Data
```bash
python src/preprocess.py --label new_command
```

### 3. Retrain Model
```bash
python src/train.py --epochs 100
```

### 4. Test Real-time
```bash
python src/infer.py
```

### 5. Evaluate Performance
```bash
python src/test_pipeline.py
```

## Support

For issues in src/:
- Check error messages in logs
- Verify audio device connection
- Test with smaller dataset first
- Check GPU/CPU availability

---

Last Updated: December 25, 2024
Version: 2.0.0
Status: Production Ready
Maintained By: Empel06
