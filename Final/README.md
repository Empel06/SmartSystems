# Final/ - Production Deployment Code

## Location

SmartSystems/Final/ contains the optimized, production-ready application code.

## Description

This is the deployment folder. The code here:
- Runs directly on Raspberry Pi
- Delivers optimal performance
- Uses ONNX-optimized models
- Has production-ready logging and monitoring

## Structure

```
Final/
├── smart_assistant.py        # Main application
```

### Requirements.txt - DEPENDENCIES

Minimal production dependencies.

```
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
sounddevice>=0.4.6
soundfile>=0.12.0
onnxruntime>=1.16.0
paho-mqtt>=1.6.0
pyttsx3>=2.90
```

Install:
```bash
pip install -r Final/requirements_prod.txt
```

## Deployment Guide

### 1. Prepare Raspberry Pi

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
  python3.10 \
  python3-pip \
  python3-dev \
  libasound2 \
  libatlas-base-dev

# Create project directory
mkdir ~/smartsystems
cd ~/smartsystems
```

### 2. Copy Application

```bash
# Copy Final folder to Pi
scp -r Final/ pi@192.168.1.100:~/smartsystems/

# Copy models folder
scp -r models/ pi@192.168.1.100:~/smartsystems/

# SSH into Pi
ssh pi@192.168.1.100

# Verify files
ls -la ~/smartsystems/Final/
```
### 3. Test Locally

```bash
# Test audio input
python3 -c "import sounddevice; print(sounddevice.query_devices())"

# Test model loading
python3 -c "import onnxruntime; print(onnxruntime.__version__)"

# Test inference
cd ~/smartsystems
python3 Final/main_app.py --benchmark
```

## Running the Application

### Standard Run

```bash
python Final/smart_assistant.py

# Output:
# Loading configuration from config.json
# Initializing model: models/final_model.onnx
# Audio device: USB Microphone
# Ready to listen for commands...
#
# Recognized: start_timer (confidence: 0.94)
# Timer started for 25 minutes
```

## Main Loop Flow

```
Loop:
  1. Record 2-second audio chunk (441kHz, mono)
  2. Extract mel-spectrogram features (128x101)
  3. Run ONNX model inference (~100ms)
  4. Get prediction and confidence
  5. If confidence > threshold (0.7):
     - Recognize command
     - Execute action (timer, sensor, etc.)
     - Generate TTS response
     - Speak response
  6. Repeat
```

## Troubleshooting Production

### "No audio input"

```bash
# Check microphone
arecord -l
aplay -l

# Test recording
arecord -D default -f S16_LE -r 16000 test.wav

# Check volume
alsamixer
```

### "Low accuracy"

```
Solutions:
1. Check audio quality (is microphone working?)
2. Adjust confidence threshold (lower = more sensitive)
3. Retrain with more data
4. Try different acoustic environment
```

### "High latency"

```
Solutions:
1. Reduce chunk size in config.json
2. Reduce mel frequency bins (64 instead of 128)
3. Check CPU usage: top -p $(pgrep -f main_app.py)
4. Profile: python Final/main_app.py --profile
```

### "Memory leak"

```
Solutions:
1. Monitor memory: watch free -h
2. Check for infinite loops
3. Clear audio buffer
4. Restart systemd service: sudo systemctl restart smartsystems
```

### "TTS not working"

```
Solutions:
1. Check: which espeak
2. Install: sudo apt install espeak
3. Test: echo "Hello" | espeak
4. Check volume: alsamixer
```
