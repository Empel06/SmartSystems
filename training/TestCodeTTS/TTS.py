"""
TTS with espeak command line (bypasses aplay errors)
Works directly with Bluetooth speakers
"""

import os
import subprocess
import time

print("="*60)
print("TTS Test - Direct espeak (Bluetooth Edition)")
print("="*60)

def speak_dutch(text):
    """
    Speak Dutch text using espeak command
    Bypasses pyttsx3/aplay limitations
    """
    try:
        # espeak -v nl (Dutch voice)
        cmd = ['espeak', '-v', 'nl', text]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✓ Speaking: '{text}'")
    except Exception as e:
        print(f"✗ Error: {e}")

# Test 1: Simple Dutch text
print("\n" + "="*60)
print("Test 1: Simple Dutch text")
print("="*60)
speak_dutch("Hallo, dit is een test. De tekst naar spraak werkt goed.")
time.sleep(2)

# Test 2: Timer start
print("="*60)
print("Test 2: Timer start message")
print("="*60)
speak_dutch("Timer is gestart. Vijftig minuten.")
time.sleep(2)

# Test 3: Pause message
print("="*60)
print("Test 3: Timer pause message")
print("="*60)
speak_dutch("Timer staat op pauze.")
time.sleep(2)

# Test 4: Resume message
print("="*60)
print("Test 4: Timer resume message")
print("="*60)
speak_dutch("Timer is hervat.")
time.sleep(2)

# Test 5: Work time message
print("="*60)
print("Test 5: Work time message")
print("="*60)
speak_dutch("Je hebt gewerkt voor 1 uur, 25 minuten, 30 seconden")
time.sleep(2)

# Test 6: Temperature message
print("="*60)
print("Test 6: Temperature message")
print("="*60)
speak_dutch("Kamertemperatuur: 22.5 graden Celsius. Luchtvochtigheid: 55 procent. Luchtkwaliteit is goed.")
time.sleep(3)

# Test 7: Air quality warning
print("="*60)
print("Test 7: Air quality warning")
print("="*60)
speak_dutch("Waarschuwing: Luchtkwaliteit is slecht. Zet een raam open of zet de ventilatie aan.")
time.sleep(3)

# Test 8: Timer finished
print("="*60)
print("Test 8: Timer finished message")
print("="*60)
speak_dutch("Timer afgelopen. Goed werk gedaan!")
time.sleep(2)

print("="*60)
print("✓ All TTS tests completed!")
print("="*60)