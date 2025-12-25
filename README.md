# Voice-Controlled AI for Productivity Support

## Overview
This project focuses on developing a voice-controlled AI system designed to support users during study or work sessions. The AI processes spoken commands, provides feedback, and interacts naturally with the user. It can handle simple requests such as:
- “Start the timer”
- “temperature?”
- “Pause timer”

A future extension includes multilingual speech recognition to increase accessibility for a wider audience.

### Key Features
- **Voice Command Recognition** - Understands natural speech input
- **Smart Timers** - Start, pause
- **Environmental Monitoring** - Real-time temperature and room status reporting
- **Local AI Processing** - Runs entirely on-device for low latency and privacy
- **Natural Interaction** - Voice synthesis for conversational feedback

### Supported Commands
```
"Start"
"Pauze"
"Time"
"Temperatuur"
```

## Problem Statement
Many students and remote workers struggle to maintain concentration for long periods. Environmental factors such as noise, temperature, or irregular breaks can affect productivity.

While separate tools exist—such as timers, environment monitors, or digital assistants—there is no integrated, voice-operated solution tailored specifically to productivity and focus. This project aims to fill that gap by creating a speech-driven interface that responds in real time and assists the user in staying focused.

## Research Question
How can we develop a voice-controlled AI that communicates naturally with users, understands basic spoken commands, and improves focus and efficiency during study or work?

## Materials and Methods

### Hardware
- Raspberry Pi 4
- Microphone (speech input)  
- HD Monitor (visual output)  
- Bleuthooth speaker

### Software
- **Language**: Python 3.10+
- **ML Framework**: PyTorch 2.0+
- **Audio Processing**: librosa, soundfile, sounddevice
- **Speech**: TTS (Text-to-Speech) synthesis

### Connectivity
- Wi-Fi or Bluetooth  
- External data sources via HTTP or web APIs

### Key Libraries
```python
torch==2.0.0
torchaudio==2.0.0
librosa==0.10.0
sounddevice==0.4.6
soundfile==0.12.0
paho-mqtt==1.6.0
onnxruntime==1.16.0
numpy==1.24.0
scipy==1.10.0
```

## Project Structure

```
SmartSystems/
├── src/                          # Core Application Code
│   ├── record.py                 # Audio recording utility
│   ├── train.py                  # Model training script
│   ├── infer.py                  # Real-time inference
│   ├── preprocess.py             # Audio preprocessing
│   ├── augment.py                # Data augmentation
│   ├── commands.py               # Command definitions
│   ├── config.py                 # Configuration management
│   ├── utils.py                  # Utility functions
│   ├── smart_assistant.py        # Main AI logic
│
├── training/                     # Model Training Pipeline
│   ├── TestCodeTTS/              # TTS synchronization fix 
│   ├── extract_features.py       # Feature extraction
│   └── TestCodeSensor/           # Sensor testing
│
├── models/                       # Trained Models
│   └── kws_cnn.pt                # Keyword spotting model
│
├── dataset/                      # Training Data
│   ├── preprocessed/             # Processed features
│   ├── pause_timer/              # Training samples
│   ├── start_timer/              # Training samples
│   ├── room_temp/                # Training samples
│   └── silence/                  # Silence samples
│
├── Final/                        # Production Code
│   ├── smart_assistant.py        # Final application
│
├── Media/                        # Documentation Assets
│   └── smartsystems.png          # Project visual
│
├── Docs/                         # Documentation
│   └── Projectvoorstel.docx      # Project proposal
│
├── venv/                         # Virtual Environment
│   ├── lib/site-packages/        # Dependencies
│   └── Scripts/                  # Activation scripts
│
├── README.md                     # This file (Project Overview)
└── requirements.txt              # Python Dependencies
```

## Quick Start

### Approach
The speech AI system is developed to run locally with low latency and the possibility for offline use. The Jetson Nano processes audio input, interprets spoken commands, and returns appropriate visual or audio feedback.

### 1. **Setup Environment**

```bash
# Clone repository
git clone https://github.com/yourusername/SmartSystems.git
cd SmartSystems

# Install dependencies
pip install -r requirements.txt
```

### 2. **Record Training Data**

```bash
# Record samples for each command (±30 per command)
python src/record.py --label start_timer --n 30
python src/record.py --label pause_timer --n 30
python src/record.py --label room_temp --n 30
python src/record.py --label silence --n 20
```

3. Controleer je dataset

Controleer of de mappen en .wav-bestanden correct zijn opgeslagen in dataset/<label>/.

### 3. **Preprocess Audio**

```bash
# Generate log-mel spectrogram features
python src/preprocess.py
```

Output: `dataset/preprocessed/<label>.npy`

### 4. **Train Model**

```bash
# Train the keyword spotting model
python src/train.py
```

The model will be saved as:
```bash
models/kws_cnn.pt
```

Watch the training loss en validation accuracy in the terminal.

Note:
With ±30 samples a label is the accuracy not perfect, but for small-vocabulary command recognition enough. Add more samples or augmentation [Augment.py](./src/Augment.py) to improve accuracy.

### 5. **Real-time Testing**

```bash
# Start the real-time keyword spotter
python src/infer.py
```

When you speak a command, you'll see:
```
Recognized: start_timer (score: 0.94)
```

## Documentation by Folder

| Folder | Purpose | README |
|--------|---------|--------|
| `/src/` | Core application modules | [src/README.md](src/README.md) |
| `/training/` | Model training pipeline | [training/README.md](training/README.md) |
| `/models/` | Trained models & inference | [models/README.md](models/README.md) |
| `/dataset/` | Training data management | [dataset/README.md](dataset/README.md) |
| `/Final/` | Production deployment code | [Final/README.md](Final/README.md) |

## Expected Results
The final prototype will be able to:
- Start and manage timers  
- Report elapsed work time  
- Pause timers  
- Provide environmental information such as room temperature (when linked to sensors or APIs)

The project aims to create a natural and intuitive voice interface that enhances productivity and simplifies interaction during study or work.

Future improvements include adding support for multiple languages to broaden usability.

## References
1. Speech & Text  
2. Medium  
3. GitHub – Uberi/SpeechRecognition  
4. GeeksForGeeks
