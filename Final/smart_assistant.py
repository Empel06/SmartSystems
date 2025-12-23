# src/smart_assistant.py
"""
Smart Voice Assistant: Timer, Temperature/Air Quality, Work Time Tracker
Integrates keyword spotting with real-time control logic
Uses AHT21 (temp/humidity) + ENS160 (air quality) sensors
"""

import os
import time
import subprocess
import tempfile
from datetime import datetime, timedelta
from enum import Enum

import torch
import numpy as np
import librosa
import soundfile as sf
from scipy.special import softmax

# Sensor imports
try:
    import board
    import busio
    import adafruit_ens160
    import adafruit_ahtx0
    SENSORS_AVAILABLE = True
except ImportError:
    print("Warning: Sensor libraries not available (dev mode)")
    SENSORS_AVAILABLE = False

# Import model from training script
from train import SimpleCNN

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "models/kws_cnn.pt"
SAMPLE_RATE = 44100
DURATION = 2.0
CONFIDENCE_THRESHOLD = 0.60

# Timer
TIMER_DURATION = 50 * 60  # 50 minutes in seconds (Pomodoro)

# Air Quality Thresholds
AQI_WARNING_THRESHOLD = 3  # AQI >= 3 = poor air quality
TVOC_WARNING_THRESHOLD = 500  # ppb

# Auto air quality check interval (seconds)
AIR_QUALITY_CHECK_INTERVAL = 30  # Check every 30 seconds

# ============================================================================
# SENSOR INITIALIZATION
# ============================================================================

class SensorManager:
    """Manages AHT21 + ENS160 sensors"""
    
    def __init__(self):
        self.aht = None
        self.ens = None
        self.initialized = False
        
        if SENSORS_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.aht = adafruit_ahtx0.AHTx0(i2c)
                self.ens = adafruit_ens160.ENS160(i2c)
                self.initialized = True
                print("Sensors initialized (AHT21 + ENS160)")
            except Exception as e:
                print(f"Sensor init error: {e}")
    
    def read_environment(self):
        """Read temperature, humidity, and air quality"""
        if not self.initialized:
            return None, None, None, None, None
        
        try:
            # Read AHT21
            temperature = self.aht.temperature
            humidity = self.aht.relative_humidity
            
            # Update ENS160 with environment data
            self.ens.temperature_compensation = temperature
            self.ens.humidity_compensation = humidity
            
            # Read ENS160
            aqi = self.ens.AQI      # Air Quality Index
            tvoc = self.ens.TVOC    # ppb
            eco2 = self.ens.eCO2    # ppm
            
            return temperature, humidity, aqi, tvoc, eco2
        
        except Exception as e:
            print(f"Sensor read error: {e}")
            return None, None, None, None, None
    
    def check_air_quality(self) -> tuple:
        """
        Check if ventilation is needed
        Returns: (needs_ventilation, aqi, tvoc)
        """
        temp, humidity, aqi, tvoc, eco2 = self.read_environment()
        
        if temp is None:
            return False, None, None
        
        needs_ventilation = aqi >= AQI_WARNING_THRESHOLD or tvoc >= TVOC_WARNING_THRESHOLD
        return needs_ventilation, aqi, tvoc

# ============================================================================
# TIMER & STATE MANAGEMENT
# ============================================================================

class TimerState(Enum):
    STOPPED = 0
    RUNNING = 1
    PAUSED = 2

class SmartAssistant:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.labels = None
        self.sensors = SensorManager()
        self.load_model()
        
        # Timer state
        self.timer_state = TimerState.STOPPED
        self.timer_duration = TIMER_DURATION  # 50 min
        self.timer_start_time = None
        self.timer_pause_time = None
        self.timer_paused_duration = 0
        self.timer_finished = False
        
        # Work tracking (accumulates across multiple sessions)
        self.total_work_time = 0  # Total seconds worked (all sessions)
        self.session_start_time = None
        self.session_paused_duration = 0
        
        # Air quality tracking
        self.last_air_quality_check = 0
        self.last_air_quality_warning = False
        
        print("Smart Assistant initialized")
        print(f"  Device: {self.device}")
        print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
        print(f"  Timer duration: {TIMER_DURATION // 60} minutes")
    
    # ========================================================================
    # MODEL LOADING
    # ========================================================================
    
    def load_model(self):
        """Load trained keyword spotting model"""
        if not os.path.exists(MODEL_PATH):
            print(f"Model not found at {MODEL_PATH}")
            return False
        
        ckpt = torch.load(MODEL_PATH, map_location=self.device)
        self.labels = ckpt["labels"]
        num_classes = len(self.labels)
        
        self.model = SimpleCNN(in_ch=1, num_classes=num_classes).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        
        print(f"Model loaded")
        print(f"  Labels: {self.labels}")
        return True
    
    # ========================================================================
    # AUDIO PROCESSING
    # ========================================================================
    
    def extract_log_mel(self, y, sr=SAMPLE_RATE, n_mels=40, hop_length=160, n_fft=512):
        """Extract log-mel spectrogram from audio"""
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        log_mel = librosa.power_to_db(mel)
        return log_mel
    
    def record_audio(self, duration=DURATION):
        """Record audio using arecord"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        
        cmd = [
            "arecord",
            "-D", "hw:3,0",  # USB mic
            "-f", "S16_LE",
            "-r", str(SAMPLE_RATE),
            "-d", str(int(duration)),
            "-c", "1",
            wav_path,
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return wav_path
        except Exception as e:
            print(f"Recording error: {e}")
            return None
    
    def recognize_command(self) -> tuple:
        """
        Record and recognize spoken command
        Returns: (label, confidence_score)
        """
        print("Recording...")
        wav_path = self.record_audio()
        
        if wav_path is None:
            return None, 0.0
        
        try:
            # Load and preprocess audio
            y, sr = sf.read(wav_path)
            os.remove(wav_path)
            
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            
            # Skip if too quiet
            if np.max(np.abs(y)) < 0.01:
                print("  [Silence detected, skipping]")
                return None, 0.0
            
            # Extract features and infer
            feats = self.extract_log_mel(y)
            x = torch.tensor(feats).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                out = self.model(x)
                probs = softmax(out.cpu().numpy()[0])
                idx = int(np.argmax(probs))
                score = float(probs[idx])
                label = self.labels[idx]
            
            # Apply confidence threshold
            if score < CONFIDENCE_THRESHOLD:
                print(f"  [Uncertain: {label} {score:.2f}, threshold {CONFIDENCE_THRESHOLD:.0%}]")
                return None, 0.0
            
            return label, score
        
        except Exception as e:
            print(f"Inference error: {e}")
            return None, 0.0
    
    # ========================================================================
    # TIMER CONTROL (50 minutes)
    # ========================================================================
    
    def start_timer(self):
        """Start a 50-minute timer (or resume if paused)"""
        if self.timer_state == TimerState.RUNNING:
            print("Timer already running!")
            return
        
        # If paused, just resume instead of starting new
        if self.timer_state == TimerState.PAUSED:
            self.resume_timer()
            return
        
        self.timer_start_time = time.time()
        self.timer_pause_time = None
        self.timer_paused_duration = 0
        self.timer_state = TimerState.RUNNING
        self.timer_finished = False
        
        print(f"Timer started: 50:00 (Pomodoro mode)")
        self._play_sound("start")
    
    def resume_timer(self):
        """Resume a paused timer"""
        if self.timer_state != TimerState.PAUSED:
            return
        
        # Calculate pause duration and continue
        pause_duration = time.time() - self.timer_pause_time
        self.timer_paused_duration += pause_duration
        self.timer_pause_time = None
        self.timer_state = TimerState.RUNNING
        print("Timer resumed")
    
    def pause_timer(self):
        """Pause the timer"""
        if self.timer_state == TimerState.STOPPED:
            print("No timer running!")
            return
        
        if self.timer_state == TimerState.RUNNING:
            self.timer_pause_time = time.time()
            self.timer_state = TimerState.PAUSED
            print("Timer paused")
        elif self.timer_state == TimerState.PAUSED:
            print("Timer already paused")
    
    def stop_timer(self):
        """Stop the timer"""
        if self.timer_state == TimerState.STOPPED:
            return
        
        self.timer_state = TimerState.STOPPED
        self.timer_start_time = None
        self.timer_pause_time = None
        print("Timer stopped")
    
    def get_timer_remaining(self) -> float:
        """Get remaining time in seconds"""
        if self.timer_state == TimerState.STOPPED:
            return self.timer_duration
        
        if self.timer_state == TimerState.PAUSED:
            elapsed = self.timer_pause_time - self.timer_start_time - self.timer_paused_duration
        else:
            elapsed = time.time() - self.timer_start_time - self.timer_paused_duration
        
        remaining = max(0, self.timer_duration - elapsed)
        return remaining
    
    def get_timer_display(self) -> str:
        """Get current timer display"""
        remaining = self.get_timer_remaining()
        mins, secs = divmod(int(remaining), 60)
        
        if self.timer_state == TimerState.STOPPED:
            return "[Stopped]"
        elif self.timer_state == TimerState.RUNNING:
            return f"{mins:02d}:{secs:02d}"
        else:  # PAUSED
            return f"(PAUSED) {mins:02d}:{secs:02d}"
    
    def check_timer_finished(self):
        """Check if timer finished and add to work time"""
        if self.timer_state == TimerState.RUNNING and not self.timer_finished:
            remaining = self.get_timer_remaining()
            
            if remaining <= 0:
                self.timer_finished = True
                self.timer_state = TimerState.STOPPED
                
                # Add session time to total work time
                session_duration = self.timer_duration
                self.total_work_time += session_duration
                
                print("\n" + "="*60)
                print("TIMER FINISHED! 50 minutes completed!")
                self.announce_work_time()
                print("="*60)
                self._play_sound("finish")
    
    def _play_sound(self, sound_type):
        """Play notification sound (optional)"""
        try:
            if sound_type == "start":
                os.system("beep -f 1000 -l 100")
            elif sound_type == "finish":
                os.system("beep -f 800 -l 200 && beep -f 800 -l 200")
        except:
            pass  # Beep not available
    
    # ========================================================================
    # TEMPERATURE & AIR QUALITY
    # ========================================================================
    
    def check_and_announce_air_quality(self, force=False):
        """
        Automatically check air quality periodically
        Or announce if explicitly requested
        """
        current_time = time.time()
        time_since_check = current_time - self.last_air_quality_check
        
        # Only check if enough time has passed (unless forced)
        if not force and time_since_check < AIR_QUALITY_CHECK_INTERVAL:
            return
        
        self.last_air_quality_check = current_time
        
        needs_ventilation, aqi, tvoc = self.sensors.check_air_quality()
        
        if needs_ventilation:
            # Only warn if we haven't warned recently
            if not self.last_air_quality_warning:
                print("\n" + "!"*60)
                print("WARNING: Air quality is poor!")
                print(f"AQI: {aqi} | TVOC: {tvoc} ppb")
                print("Open a window or turn on ventilation!")
                print("!"*60 + "\n")
                self._play_sound("start")
                self.last_air_quality_warning = True
        else:
            self.last_air_quality_warning = False
    
    def announce_environment(self):
        """Read and announce temperature + air quality (on demand)"""
        temp, humidity, aqi, tvoc, eco2 = self.sensors.read_environment()
        
        if temp is None:
            print("Could not read sensors")
            return
        
        print(f"\nTemperature: {temp:.1f}C")
        print(f"Humidity: {humidity:.1f}%")
        print(f"Air Quality Index: {aqi}")
        print(f"TVOC: {tvoc} ppb  |  eCO2: {eco2} ppm")
        
        # Check if ventilation needed
        if aqi >= 3 or tvoc >= TVOC_WARNING_THRESHOLD:
            print("\nWARNING: Air quality is poor!")
            print("   Open a window or turn on ventilation")
            self._play_sound("start")
        else:
            print("Air quality is good")
        print()
    
    # ========================================================================
    # WORK TIME TRACKING (ACCUMULATES)
    # ========================================================================
    
    def start_work_session(self):
        """Start tracking a new work session"""
        if self.session_start_time is not None:
            print("Work session already active!")
            return
        
        self.session_start_time = time.time()
        self.session_paused_duration = 0
        print("Work session started")
    
    def get_current_session_time(self) -> int:
        """Get elapsed time in current session"""
        if self.session_start_time is None:
            return 0
        
        elapsed = time.time() - self.session_start_time - self.session_paused_duration
        return int(elapsed)
    
    def get_total_work_time_display(self) -> str:
        """Get total accumulated work time (all sessions)"""
        # Current session time
        current_session = self.get_current_session_time()
        total = self.total_work_time + current_session
        
        hours, remainder = divmod(total, 3600)
        mins, secs = divmod(remainder, 60)
        
        return f"Total worked: {hours:02d}:{mins:02d}:{secs:02d}"
    
    def announce_work_time(self):
        """Announce total work time across all sessions"""
        display = self.get_total_work_time_display()
        print(display)
    
    def end_work_session(self):
        """End current session (called when timer finishes)"""
        if self.session_start_time is None:
            return
        
        session_time = self.get_current_session_time()
        self.total_work_time += session_time
        self.session_start_time = None
        print(f"Session ended. Total work time: {self.get_total_work_time_display()}")
    
    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    
    def process_command(self, label: str, score: float):
        """Execute action based on recognized command"""
        print(f"Recognized: {label} ({score:.2f})")
        
        if label == "start":
            self.start_work_session()
            self.start_timer()
        
        elif label == "pauze":
            self.pause_timer()
        
        elif label == "room_temp":
            self.announce_environment()
        
        elif label == "work_time":
            self.announce_work_time()
        
        elif label == "silence":
            pass  # Ignore silence
        
        else:
            print(f"Unknown command: {label}")
    
    def run(self):
        """Main listening loop"""
        print("\n" + "="*60)
        print("Smart Voice Assistant Started")
        print("="*60)
        print("Commands:")
        print("  'start'      - Start or resume 50-minute timer")
        print("  'pauze'      - Pause timer")
        print("  'room_temp'  - Check temperature & air quality")
        print("  'work_time'  - Show total work time")
        print("\nAir quality is checked automatically every 30 seconds")
        print("Press CTRL+C to stop\n")
        
        try:
            while True:
                # Check if timer finished
                self.check_timer_finished()
                
                # Auto-check air quality (every 30 seconds)
                self.check_and_announce_air_quality(force=False)
                
                # Listen for command
                label, score = self.recognize_command()
                
                if label is not None:
                    self.process_command(label, score)
                
                # Show status
                print(f"  {self.get_timer_display()} | {self.get_total_work_time_display()}")
                print()
        
        except KeyboardInterrupt:
            print("\nStopped")
            self.stop_timer()
            self.end_work_session()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    assistant = SmartAssistant()
    assistant.run()