"""
Simple TTS Test Script
Test if pyttsx3 works on your Raspberry Pi
"""
import pyttsx3
import time

print("="*60)
print("TTS (Text-to-Speech) Test")
print("="*60)

# Initialize TTS engine
try:
    engine = pyttsx3.init()
    print("TTS Engine initialized successfully")
except Exception as e:
    print(f"Error initializing TTS: {e}")
    exit(1)

# Get all available voices
voices = engine.getProperty('voices')
print(f"\nAvailable voices: {len(voices)}")
for i, voice in enumerate(voices):
    print(f"  {i}: {voice.name} (ID: {voice.id})")
    print(f"     Languages: {voice.languages}")

# Set speech rate and volume
engine.setProperty('rate', 150)  # Speed (words per minute)
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

print(f"\nSpeech rate set to 150 WPM")
print(f"Volume set to 0.9")

# Test 1: Simple Dutch text
print("\n" + "="*60)
print("Test 1: Simple Dutch text")
print("="*60)
text1 = "Hallo, dit is een test. De tekst naar spraak werkt goed."
print(f"Speaking: '{text1}'")
engine.say(text1)
engine.runAndWait()
print("Test 1 complete\n")

time.sleep(1)

# Test 2: Timer start
print("="*60)
print("Test 2: Timer start message")
print("="*60)
text2 = "Timer is gestart. Vijftig minuten."
print(f"Speaking: '{text2}'")
engine.say(text2)
engine.runAndWait()
print("Test 2 complete\n")

time.sleep(1)

# Test 3: Pause message
print("="*60)
print("Test 3: Timer pause message")
print("="*60)
text3 = "Timer staat op pauze."
print(f"Speaking: '{text3}'")
engine.say(text3)
engine.runAndWait()
print("Test 3 complete\n")

time.sleep(1)

# Test 4: Resume message
print("="*60)
print("Test 4: Timer resume message")
print("="*60)
text4 = "Timer is hervat."
print(f"Speaking: '{text4}'")
engine.say(text4)
engine.runAndWait()
print("Test 4 complete\n")

time.sleep(1)

# Test 5: Work time message
print("="*60)
print("Test 5: Work time message")
print("="*60)
text5 = "Je hebt gewerkt voor 1 uur, 25 minuten, 30 seconden"
print(f"Speaking: '{text5}'")
engine.say(text5)
engine.runAndWait()
print("Test 5 complete\n")

time.sleep(1)

# Test 6: Temperature message
print("="*60)
print("Test 6: Temperature message")
print("="*60)
text6 = "Kamertemperatuur: 22.5 graden Celsius. Luchtvochtigheid: 55 procent. Luchtkwaliteit is goed."
print(f"Speaking: '{text6}'")
engine.say(text6)
engine.runAndWait()
print("Test 6 complete\n")

time.sleep(1)

# Test 7: Air quality warning
print("="*60)
print("Test 7: Air quality warning")
print("="*60)
text7 = "Waarschuwing: Luchtkwaliteit is slecht. Zet een raam open of zet de ventilatie aan."
print(f"Speaking: '{text7}'")
engine.say(text7)
engine.runAndWait()
print("Test 7 complete\n")

time.sleep(1)

# Test 8: Timer finished
print("="*60)
print("Test 8: Timer finished message")
print("="*60)
text8 = "Timer afgelopen. Goed werk gedaan!"
print(f"Speaking: '{text8}'")
engine.say(text8)
engine.runAndWait()
print("Test 8 complete\n")

print("="*60)
print("All TTS tests completed successfully!")
print("="*60)