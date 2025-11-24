# Voice-Controlled AI for Productivity Support

## Overview
This project focuses on developing a voice-controlled AI system designed to support users during study or work sessions. The AI processes spoken commands, provides feedback, and interacts naturally with the user. It can handle simple requests such as:
- “Start the timer”
- “How warm is it?”
- “Pause the Pomodoro timer”

A future extension includes multilingual speech recognition to increase accessibility for a wider audience.

## Problem Statement
Many students and remote workers struggle to maintain concentration for long periods. Environmental factors such as noise, temperature, or irregular breaks can affect productivity.

While separate tools exist—such as timers, environment monitors, or digital assistants—there is no integrated, voice-operated solution tailored specifically to productivity and focus. This project aims to fill that gap by creating a speech-driven interface that responds in real time and assists the user in staying focused.

## Research Question
How can we develop a voice-controlled AI that communicates naturally with users, understands basic spoken commands, and improves focus and efficiency during study or work?

## Materials and Methods

### Hardware
- NVIDIA Jetson Nano (processing unit) or Raspberry Pi 4
- Microphone (speech input)  
- HD Monitor (visual output)  
- Button (physical input)

### Software
- Python  
- PyTorch  
- Optional C++ for performance-critical functions  
- SpeechRecognition (Uberi), open-source language models, and multilingual speech-to-text systems

### Connectivity
- Wi-Fi or Bluetooth  
- External data sources via HTTP or web APIs

### Approach
The speech AI system is developed to run locally with low latency and the possibility for offline use. The Jetson Nano processes audio input, interprets spoken commands, and returns appropriate visual or audio feedback.

```bash
python src/record.py --label start_timer --n 30
python src/record.py --label pause_timer --n 30
python src/record.py --label room_temp --n 30
python src/record.py --label pomodoro --n 30
# optioneel:
python src/record.py --label silence --n 20
```

3. Controleer je dataset

Controleer of de mappen en .wav-bestanden correct zijn opgeslagen in dataset/<label>/.

## Audio Preprocessing
```bash
python src/preprocess.py
```

Dit genereert:
```bash
dataset/preprocessed/<label>.npy
```
met log-mel spectrogram features.

## Model Trainen
Train het model vanuit de project root:
```bash
python src/train.py
```

Het model wordt opgeslagen als:
```bash
models/kws_cnn.pt
```

Let op de training loss en validation accuracy in de terminal.

Opmerking:
Met ±30 samples per label is de accuracy niet perfect, maar voor small-vocabulary command recognition vaak voldoende. Voeg meer samples of augmentatie toe om de accuracy te verbeteren.

## Realtime Testen
Start de realtime keyword spotter:
```bash
python src/infer.py
```

Wanneer je een commando inspreekt, verschijnt:
```bash
Recognized: <label> (score)
```

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
