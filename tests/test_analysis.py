"""Tests that build small audio signals with a known answer.

Run them with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import (  # noqa: E402
    NO_KEY,
    SAMPLE_RATE,
    analyze,
    detect_key,
    detect_tempo,
    relative_key,
)

A4 = 440.0


def note_frequency(semitones_from_a4: int) -> float:
    return A4 * 2 ** (semitones_from_a4 / 12)


def make_clicks(bpm: float, seconds: float = 20.0) -> np.ndarray:
    """A metronome: one short click on every beat."""
    y = np.zeros(int(seconds * SAMPLE_RATE))
    click_time = np.arange(int(0.03 * SAMPLE_RATE)) / SAMPLE_RATE
    click = np.sin(2 * np.pi * 1000 * click_time) * np.exp(-click_time * 150)
    for beat in np.arange(0, seconds - 0.1, 60.0 / bpm):
        start = int(beat * SAMPLE_RATE)
        y[start : start + len(click)] += click
    return y


def make_chords(chords: list[list[int]], seconds_per_chord: float = 2.0) -> np.ndarray:
    """Play chords one after another. Notes are semitones from A4."""
    t = np.arange(int(seconds_per_chord * SAMPLE_RATE)) / SAMPLE_RATE
    parts = []
    for chord in chords:
        sound = sum(np.sin(2 * np.pi * note_frequency(n) * t) for n in chord)
        parts.append(sound / len(chord))
    return np.concatenate(parts) * 0.5


# Chords as semitone offsets from A4: C4 = -9, D4 = -7, E4 = -5, F4 = -4,
# G4 = -2, A4 = 0, B4 = 2, C5 = 3.
C_MAJOR = [-9, -5, -2]
F_MAJOR = [-4, 0, 3]
G_MAJOR = [-2, 2, 5]
A_MINOR = [-12, -9, -5]
D_MINOR = [-7, -4, 0]
E_MAJOR = [-17, -13, -10]  # the dominant chord that gives A minor its G#


def make_half_time_drums(bpm: float, bars: int = 12) -> np.ndarray:
    """A trap-style bar: kicks on beats 1 and 3.5, snare on 3, 8th-note hats.

    The off-beat kick is what fools simple beat trackers into answering
    2/3 of the real tempo.
    """
    rng = np.random.default_rng(0)
    beat = 60.0 / bpm
    y = np.zeros(int(bars * 4 * beat * SAMPLE_RATE) + SAMPLE_RATE)

    def place(sound: np.ndarray, at_beat: float) -> None:
        start = int(at_beat * beat * SAMPLE_RATE)
        y[start : start + len(sound)] += sound

    t = np.arange(int(0.25 * SAMPLE_RATE)) / SAMPLE_RATE
    kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-t * 30)) * t) * np.exp(-t * 14)
    snare = rng.normal(0, 0.6, t.size) * np.exp(-t * 25)
    hat = rng.normal(0, 0.25, t.size) * np.exp(-t * 90)

    for bar in range(bars):
        first_beat = bar * 4
        place(kick, first_beat)
        place(kick, first_beat + 2.5)
        place(snare, first_beat + 2)
        for eighth in range(8):
            place(hat, first_beat + eighth / 2)
    return y


def closest_reading(readings: tuple[float, float], bpm: float) -> float:
    return min(readings, key=lambda value: abs(value - bpm))


@pytest.mark.parametrize("bpm", [90, 120, 140])
def test_tempo_of_a_metronome(bpm):
    readings = detect_tempo(make_clicks(bpm), SAMPLE_RATE)
    assert closest_reading(readings, bpm) == pytest.approx(bpm, abs=0.5)


@pytest.mark.parametrize("bpm", [140, 150, 160])
def test_tempo_of_half_time_drums(bpm):
    readings = detect_tempo(make_half_time_drums(bpm), SAMPLE_RATE)
    assert closest_reading(readings, bpm) == pytest.approx(bpm, abs=0.5)


def test_relative_key():
    assert relative_key("C major") == "A minor"
    assert relative_key("F# minor") == "A major"
    assert relative_key(relative_key("A# minor")) == "A# minor"


def test_c_major_progression():
    y = make_chords([C_MAJOR, F_MAJOR, G_MAJOR, C_MAJOR] * 2)
    key, confidence, _ = detect_key(y, SAMPLE_RATE)
    assert key == "C major"
    assert confidence > 0.5


def test_a_minor_progression():
    y = make_chords([A_MINOR, D_MINOR, E_MAJOR, A_MINOR] * 2)
    key, _, _ = detect_key(y, SAMPLE_RATE)
    assert key == "A minor"


def test_analyze_reads_a_file(tmp_path):
    path = tmp_path / "chords.wav"
    sf.write(path, make_chords([C_MAJOR, F_MAJOR, G_MAJOR, C_MAJOR]), SAMPLE_RATE)
    result = analyze(str(path))
    assert result.duration_seconds == pytest.approx(8.0, abs=0.1)
    assert result.key == "C major"


def test_metronome_has_no_key():
    key, confidence, _ = detect_key(make_clicks(120, seconds=10), SAMPLE_RATE)
    assert key == NO_KEY
    assert confidence == 0.0


def test_drums_have_no_key():
    key, _, _ = detect_key(make_half_time_drums(140), SAMPLE_RATE)
    assert key == NO_KEY


def test_silence_is_rejected(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(SAMPLE_RATE * 2), SAMPLE_RATE)
    with pytest.raises(ValueError):
        analyze(str(path))
