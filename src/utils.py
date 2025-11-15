import librosa, numpy as np

def load_audio(path, sr=16000):
    return librosa.load(path, sr=sr)[0]

def extract_features(audio, sr=16000):
    mel=librosa.feature.melspectrogram(audio, sr=sr)
    return librosa.power_to_db(mel)
