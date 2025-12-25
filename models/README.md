# models/ - Trained Models

## Location

SmartSystems/models/ contains all trained machine learning models.

## Description

This folder contains the following models:
- kws_cnn.pt - Keyword spotting CNN model

## Structure

```
models/
├── kws_cnn.pt              # CNN keyword spotter
```

## Model Files Explained

### kws_cnn.pt - KEYWORD SPOTTING MODEL

Specialized CNN for keyword spotting.

Architecture:
```
Input (1, 128, 101)
  ↓
Conv2d(32, 3x3) -> ReLU -> MaxPool(2x2)
  ↓
Conv2d(64, 3x3) -> ReLU -> MaxPool(2x2)
  ↓
Conv2d(128, 3x3) -> ReLU -> AdaptiveAvgPool
  ↓
Flatten
  ↓
Dense(256) -> ReLU -> Dropout(0.3)
  ↓
Dense(num_classes) -> Softmax
```

Usage:
```python
# Load KWS model
kws_model = torch.load('models/kws_cnn.pt')
kws_model.eval()

# Inference similar to final_model.pt
```

## Model Performance

### Accuracy Metrics (Test Set)

```
start_timer:  96% accuracy, 0.96 F1-score
pause_timer:  94% accuracy, 0.94 F1-score
room_temp:    93% accuracy, 0.93 F1-score
pomodoro:     96% accuracy, 0.96 F1-score
silence:      95% accuracy, 0.95 F1-score

Overall: 95.2% accuracy
```

### Hardware Compatibility

```
Raspberry Pi 4 (ARM32)
Raspberry Pi 5 (ARM32)
NVIDIA Jetson Nano
NVIDIA Jetson Orin
Intel x86/x64
GPU (CUDA) - if available
TPU (with conversion)
```

## Troubleshooting

### "Model not found"

```bash
# Check if model exists
ls -lh models/kws_cnn.pt

# If missing, train new model
python training/train.py
```

### "Low confidence on predictions"

```
Solutions:
1. Model may be underfitted - retrain with more epochs
2. Input features incorrect - check preprocessing
3. Model not trained well - increase training data
```
