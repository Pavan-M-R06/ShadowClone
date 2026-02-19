"""
Generate sound effects for Shadow Clone AR.
Run this script once before launching the main application.
Creates: assets/sounds/success.wav and assets/sounds/poof.wav
"""

import wave
import struct
import math
import random
import os


def generate_success_sound(filename):
    """
    Generate a rising 'jutsu activation' chime.
    Three ascending notes with harmonics — sounds like a power-up.
    """
    sample_rate = 44100
    samples = []

    # Three ascending notes: C5 → E5 → G5 (triumphant major chord arpeggio)
    notes = [
        (523.25, 0.25),   # C5 - 0.25s
        (659.25, 0.25),   # E5 - 0.25s
        (783.99, 0.45),   # G5 - 0.45s (hold the final note longer)
    ]

    for freq, duration in notes:
        num_samples = int(sample_rate * duration)
        for i in range(num_samples):
            t = i / sample_rate
            # Envelope: quick attack, sustained, gentle release
            if t < 0.02:
                envelope = t / 0.02  # 20ms attack
            elif t > duration - 0.05:
                envelope = (duration - t) / 0.05  # 50ms release
            else:
                envelope = 1.0

            # Main tone + harmonics for richness
            sample = 0.6 * math.sin(2 * math.pi * freq * t)           # fundamental
            sample += 0.25 * math.sin(2 * math.pi * freq * 2 * t)     # 2nd harmonic
            sample += 0.10 * math.sin(2 * math.pi * freq * 3 * t)     # 3rd harmonic
            sample += 0.05 * math.sin(2 * math.pi * freq * 4 * t)     # 4th harmonic

            sample *= envelope
            samples.append(int(sample * 14000))

    # Add a brief shimmering tail
    shimmer_duration = 0.3
    shimmer_freq = 783.99 * 2  # One octave above final note
    for i in range(int(sample_rate * shimmer_duration)):
        t = i / sample_rate
        envelope = math.exp(-t * 6)
        sample = envelope * 0.3 * math.sin(2 * math.pi * shimmer_freq * t)
        sample += envelope * 0.15 * math.sin(2 * math.pi * shimmer_freq * 1.5 * t)
        samples.append(int(sample * 14000))

    _write_wav(filename, samples, sample_rate)
    print(f"  ✓ Created: {filename}")


def generate_poof_sound(filename):
    """
    Generate a 'poof' / smoke cloud burst sound.
    Combination of a low thump + noise burst with rapid decay.
    """
    sample_rate = 44100
    duration = 0.6
    samples = []

    random.seed(42)  # Reproducible

    for i in range(int(sample_rate * duration)):
        t = i / sample_rate

        # Low frequency thump (sub-bass impact)
        thump_envelope = math.exp(-t * 12)
        thump = thump_envelope * 0.5 * math.sin(2 * math.pi * 60 * t)

        # Mid-frequency body
        mid_envelope = math.exp(-t * 8)
        mid = mid_envelope * 0.3 * math.sin(2 * math.pi * 150 * t)

        # High-frequency noise (the "air" sound)
        noise_envelope = math.exp(-t * 10)
        noise = noise_envelope * 0.4 * (random.random() * 2 - 1)

        # Combine
        sample = thump + mid + noise
        samples.append(int(sample * 16000))

    _write_wav(filename, samples, sample_rate)
    print(f"  ✓ Created: {filename}")


def generate_rasengan_sound(filename):
    """
    Generate a 'Rasengan' / spinning chakra ball sound.
    Continuous swirling energy: low hum + high whine with rapid modulation.
    """
    sample_rate = 44100
    duration = 3.0  # seconds
    samples = []

    for i in range(int(sample_rate * duration)):
        t = i / sample_rate

        # 1. Low hum (power core) - 120Hz oscillating slightly
        freq_low = 120 + 20 * math.sin(2 * math.pi * 2 * t)
        hum = 0.4 * math.sin(2 * math.pi * freq_low * t)

        # 2. High whine (spinning shell) - 800Hz
        freq_high = 800 + 100 * math.sin(2 * math.pi * 15 * t) # Rapid wobbling (15Hz LFO)
        whine = 0.15 * math.sin(2 * math.pi * freq_high * t)

        # 3. Turbulent noise (air displacement)
        noise = 0.1 * (random.random() * 2 - 1)

        # Apply amplitude modulation (tremolo) to simulate rotation
        tremolo = 0.8 + 0.2 * math.sin(2 * math.pi * 10 * t) # 10Hz rotation

        sample = (hum + whine + noise) * tremolo
        
        # Loop fade in/out slightly to avoid clicks if looped
        envelope = 1.0
        if t < 0.1: envelope = t / 0.1
        elif t > duration - 0.1: envelope = (duration - t) / 0.1
        
        samples.append(int(sample * 16000 * envelope))

    _write_wav(filename, samples, sample_rate)
    print(f"  ✓ Created: {filename}")


def _write_wav(filename, samples, sample_rate):
    """Write samples to a WAV file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with wave.open(filename, 'w') as wav:
        wav.setparams((1, 2, sample_rate, len(samples), 'NONE', 'not compressed'))
        for s in samples:
            clamped = max(-32768, min(32767, s))
            wav.writeframes(struct.pack('<h', clamped))


if __name__ == '__main__':
    print("🔊 Generating Shadow Clone AR sound effects...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sounds_dir = os.path.join(base_dir, 'assets', 'sounds')

    generate_success_sound(os.path.join(sounds_dir, 'success.wav'))
    generate_poof_sound(os.path.join(sounds_dir, 'poof.wav'))
    generate_rasengan_sound(os.path.join(sounds_dir, 'rasengan.wav'))

    print("\n✅ All sound effects generated!")
    print(f"   Location: {sounds_dir}")
