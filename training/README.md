# training/ - Model Training Pipeline

## Location

SmartSystems/training/ contains all training-related code for model development.

## Description

This folder is for training and optimizing machine learning models. It contains:
- Training scripts
- Feature extraction
- Model evaluation
- TTS synchronization fixes
- Sensor testing code

## Structure

```
training/
├── train.py                  # Main training script
├── TTS_paging_2.py          # TTS synchronization fix
├── extract_features.py       # Feature extraction utilities
├── evaluator.py              # Model evaluation and metrics
├── TestCodeSensor/           # Sensor testing modules
│   ├── __init__.py
│   └── test_sensors.py
└── requirements_training.txt # Training-specific dependencies
```

## Key Files

### 1. train.py - MAIN TRAINING SCRIPT

The main training script for the CNN model.

Usage:
```bash
# Standard training
python training/train.py

# With custom parameters
python training/train.py \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --dataset dataset/preprocessed

# Resume from checkpoint
python training/train.py --resume models/checkpoint.pt

# Evaluation only (no training)
python training/train.py --evaluate models/final_model.pt
```

What it does:
1. Load preprocessed audio features from dataset/preprocessed/
2. Create train/validation split (80/20)
3. Build CNN model (if not loaded from checkpoint)
4. Train with early stopping
5. Evaluate on validation set
6. Save best model
7. Convert to ONNX (optional)

Output Files:
```
models/
├── kws_cnn.pt              # Final trained model
├── best_model.pt           # Best validation score
├── final_model.onnx        # Optimized ONNX version
└── training_history.json   # Metrics and curves
```

Hyperparameters:

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| epochs | 50 | 10-200 | More = better but slower |
| batch_size | 16 | 8-64 | Smaller = slower, more stable |
| learning_rate | 0.001 | 0.0001-0.1 | Too high = divergence |
| dropout | 0.3 | 0-0.5 | Prevents overfitting |
| optimizer | Adam | Adam, SGD | Adam usually better |
| early_stop_patience | 10 | 5-20 | Stop if no improvement |

Model Architecture:
```
Input: (1, 128, 101)  # Mel-spectrogram
  ↓
Conv2d(32) -> ReLU -> MaxPool
  ↓
Conv2d(64) -> ReLU -> MaxPool
  ↓
Conv2d(128) -> ReLU -> AdaptiveAvgPool
  ↓
Flatten
  ↓
Dense(256) -> ReLU -> Dropout
  ↓
Dense(num_classes) -> Softmax
  ↓
Output: (num_classes)
```

Expected Results:
- Accuracy: 92-96% (with 30 samples/label)
- Training loss: Decreasing over time
- Validation loss: Stabilizes around epoch 30-40
- Training time: 5-10 minutes (Raspberry Pi 4)

Training Metrics:
```
Epoch 1/50: Loss=2.134, Acc=0.45
Epoch 10/50: Loss=0.523, Acc=0.87
Epoch 20/50: Loss=0.234, Acc=0.93
Epoch 30/50: Loss=0.156, Acc=0.95
Epoch 40/50: Loss=0.128, Acc=0.96
Best model saved: models/best_model.pt
```

### 2. TTS_paging_2.py - TTS SYNCHRONIZATION FIX

IMPORTANT: This file solves the overlapping TTS speech problem!

Problem:
- Multiple TTS responses could be played simultaneously
- Speech would overlap each other
- Unnatural sounding dialogue

Solution - Speech Paging Algorithm:
```python
from training.TTS_paging_2 import SpeechPager

pager = SpeechPager(
    max_concurrent_speeches=1,  # Only 1 speech at a time
    min_delay_between=0.5       # 0.5s delay between utterances
)

# Queue multiple responses
pager.queue("Timer started")
pager.queue("Duration is 25 minutes")
pager.queue("Press pause to stop")

# Speak sequentially (no overlapping)
pager.process_queue()
```

Key Features:
- Sequential speech output (no overlapping)
- Configurable delays between utterances
- FIFO queue management
- Thread-safe implementation
- Volume management
- Speed control

Usage Examples:
```python
# Simple usage
pager.say("Hello")
pager.say("How can I help?")

# With parameters
pager.say(
    "Starting timer",
    rate=150,           # Speech speed
    volume=0.9,         # Volume (0-1)
    delay_after=1.0     # Delay after speaking
)

# Batch processing
responses = [
    "Timer started",
    "Work for 25 minutes",
    "Press space to pause"
]
for response in responses:
    pager.say(response)
```

Configuration:
```python
pager.config = {
    'engine': 'pyttsx3',        # or 'gtts'
    'language': 'en',
    'speed': 150,               # Words per minute
    'volume': 1.0,              # 0-1 scale
    'min_pause': 0.3,           # Min pause (seconds)
    'max_concurrent': 1,        # Max simultaneous
}
```

Benefits:
- Natural conversation flow
- No jarring overlaps
- Professional sounding responses
- Easy to adjust timing

See Also:
- Implementation: training/TTS_paging_2.py
- Integration: src/smart_assistant.py
- Documentation: See "TTS Integration" in main README

### 3. extract_features.py - FEATURE EXTRACTION

Utility script for manual audio feature extraction.

Usage:
```bash
# Extract features for specific label
python training/extract_features.py --label start_timer

# Extract all labels
python training/extract_features.py --all

# Custom parameters
python training/extract_features.py \
  --n_mels 128 \
  --n_fft 512 \
  --hop_length 160
```

Functions:
```python
from training.extract_features import AudioFeatureExtractor

extractor = AudioFeatureExtractor(sr=16000, n_mels=128)

# Extract mel-spectrogram
features = extractor.extract_mel_spectrogram(audio_path)

# Extract MFCC
mfcc = extractor.extract_mfcc(audio_path)

# Extract multiple features
all_features = extractor.extract_all(audio_path)
```

Output Format:
```
dataset/preprocessed/
├── start_timer.npy (shape: n_samples, 128, 101)
├── pause_timer.npy
├── room_temp.npy
├── pomodoro.npy
└── silence.npy
```

### 4. evaluator.py - MODEL EVALUATION

Evaluation utilities for analyzing model performance.

Usage:
```bash
# Evaluate model on test set
python training/evaluator.py --model models/final_model.pt

# Generate detailed report
python training/evaluator.py \
  --model models/final_model.pt \
  --report detailed \
  --output evaluation_report.html

# Cross-validation (5-fold)
python training/evaluator.py --model models/final_model.pt --cv 5
```

Metrics Generated:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curves
- Per-class Performance
- Inference Time Analysis

Example Output:
```
Model Evaluation Results:
========================

Overall Metrics:
- Accuracy:  95.2%
- Precision: 94.8%
- Recall:    93.5%
- F1-Score:  94.1%

Per-Class Performance:
- start_timer:  Acc=96%, F1=0.96
- pause_timer:  Acc=94%, F1=0.94
- room_temp:    Acc=93%, F1=0.93
- pomodoro:     Acc=96%, F1=0.96
- silence:      Acc=95%, F1=0.95

Inference Performance:
- Average latency: 125ms
- Min latency: 102ms
- Max latency: 156ms
- Throughput: 8 samples/sec
```

Functions:
```python
from training.evaluator import ModelEvaluator

evaluator = ModelEvaluator(model_path='models/final_model.pt')

# Get metrics
metrics = evaluator.evaluate(test_data)

# Generate confusion matrix
cm = evaluator.confusion_matrix(test_data)

# Performance per class
per_class = evaluator.per_class_metrics(test_data)

# Inference time analysis
timing = evaluator.analyze_inference_time(test_data)
```

### 5. TestCodeSensor/ - SENSOR TESTING

Testing code for hardware sensors (temperature, humidity, etc.)

Contents:
- test_sensors.py - Sensor initialization and testing
- __init__.py - Module initialization

Usage:
```bash
# Test available sensors
python training/TestCodeSensor/test_sensors.py

# Test specific sensor
python training/TestCodeSensor/test_sensors.py --sensor temperature

# Continuous monitoring
python training/TestCodeSensor/test_sensors.py --monitor 60
```

Supported Sensors:
- Temperature (DHT22, BME680, etc.)
- Humidity (DHT22, BME680)
- Pressure (BME680)
- CO2 levels (future)

## Training Workflow

### Step 1: Prepare Data
```bash
# Record audio samples
python src/record.py --label start_timer --n 30

# Preprocess features
python src/preprocess.py
```

### Step 2: Train Model
```bash
# Run training
python training/train.py --epochs 100

# Monitor training
tail -f training/training.log
```

### Step 3: Evaluate
```bash
# Evaluate model
python training/evaluator.py --model models/final_model.pt

# Generate report
python training/evaluator.py --model models/final_model.pt --report detailed
```

### Step 4: Optimize and Deploy
```bash
# Convert to ONNX
python training/train.py --convert_onnx

# Test ONNX model
python src/infer.py --model models/final_model.onnx
```

## Training Data Structure

Expected input format:
```
dataset/preprocessed/
├── start_timer.npy
│   Shape: (30, 128, 101)  # 30 samples, 128 mel bins, 101 time frames
├── pause_timer.npy
│   Shape: (30, 128, 101)
├── room_temp.npy
│   Shape: (30, 128, 101)
├── pomodoro.npy
│   Shape: (30, 128, 101)
└── silence.npy
    Shape: (20, 128, 101)
```

## Model Versioning

```
models/
├── final_model.pt           # PyTorch format (production)
├── final_model.onnx         # ONNX format (optimized)
├── kws_cnn.pt              # Latest trained model
├── best_model.pt           # Best validation score
└── checkpoints/
    ├── epoch_10.pt
    ├── epoch_20.pt
    └── epoch_30.pt
```

## Performance Optimization

### 1. Model Compression
```bash
# ONNX conversion (4-5x smaller)
python training/train.py --convert_onnx --optimize

# Quantization
python training/train.py --quantize int8
```

### 2. Accuracy Improvement
```
Techniques:
1. More training data (100+ samples/label)
2. Data augmentation (time stretch, pitch shift)
3. Longer training (100-200 epochs)
4. Better hyperparameters (tuning)
5. Ensemble methods (multiple models)
```

### 3. Speed Improvement
```
Techniques:
1. Reduce mel frequency bins (64 instead of 128)
2. Reduce FFT size (256 instead of 512)
3. Use ONNX runtime
4. GPU acceleration (CUDA)
5. Model pruning
```

## Troubleshooting

### "Training accuracy low (70%)"
```
Solutions:
1. Record more samples per label (50-100)
2. Check audio quality
3. Use data augmentation
4. Increase training epochs
5. Adjust learning rate
```

### "Model overfitting (100% train, 70% val)"
```
Solutions:
1. Increase dropout rate (0.4-0.5)
2. Add L2 regularization
3. More training data
4. Data augmentation
5. Early stopping patience (lower)
```

### "Out of memory during training"
```
Solutions:
1. Reduce batch size (8 or 16)
2. Reduce mel frequency bins (64)
3. Reduce number of samples
4. Close other applications
5. Use swap memory
```

### "TTS speech overlapping"
```
Solutions:
- Use TTS_paging_2.py!
- Ensures sequential output
- No overlapping speeches
- See section above
```

## Dependencies

Training-specific:
```
torch>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
onnx>=1.14.0
onnxruntime>=1.16.0
```

## Learning Resources

1. Model Architecture - Understanding CNN
2. Feature Extraction - Mel-spectrogram concepts
3. Training Techniques - Early stopping, dropout, regularization
4. ONNX Optimization - Model conversion and optimization
5. Speech Processing - Audio fundamentals

## Monitoring Training

### Live Monitoring
```bash
# Terminal
python training/train.py --verbose

# TensorBoard
tensorboard --logdir=training/logs
```

### Post-Training Analysis
```bash
# Plot training curves
python training/plot_training.py

# Generate HTML report
python training/generate_report.py
```

## Best Practices

1. Always backup best model - Save before experimenting
2. Use separate validation set - Measure real performance
3. Document hyperparameters - Reproducibility
4. Version control - Track changes in git
5. Test on unseen data - Verify generalization

---

Last Updated: December 25, 2024
Version: 2.0.0
Status: Production Ready
Special Feature: TTS Synchronization Fix
