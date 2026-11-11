"""Core audio analysis: tempo (BPM) and musical key.

This file knows nothing about the user interface. app.py calls analyze()
and shows the result. Keeping logic and interface apart means we can
replace the interface later without touching the analysis.
"""

from dataclasses import dataclass

import librosa
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles: how strongly each of the 12 notes
# "belongs" to a major / minor key whose tonic sits at index 0.
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

NO_KEY = "No clear key"  # shown for drums-only audio
MIN_KEY_CONFIDENCE = 0.4  # below this the best match is too weak to trust

SAMPLE_RATE = 22050  # samples per second; plenty for tempo and key
MAX_SECONDS = 180  # analyse at most the first 3 minutes to stay fast
HOP_LENGTH = 512  # samples between two rhythm measurements (about 23 ms)

MIN_BPM = 70  # we answer inside 70-180 BPM; half / double is shown as
MAX_BPM = 180  # the alternative


@dataclass
class AnalysisResult:
    """Everything we learned about one audio file."""

    bpm: float
    alt_bpm: float  # the half-time / double-time reading of the same beat
    key: str  # best guess, e.g. "F# minor"
    relative_key: str  # the key with the same notes, e.g. "A major"
    key_confidence: float  # correlation of the best guess, from -1 to 1
    second_key: str  # runner-up
    duration_seconds: float


# ---------------------------------------------------------------- tempo


def _tempo_score(repeat: np.ndarray, frames_per_second: float, bpm: float, beats: int) -> float:
    """How well a tempo fits: average rhythm repetition after 1, 2, ... beats.

    `repeat[d]` says how similar the rhythm is to itself d frames later.
    If the tempo is right, the rhythm repeats after every whole beat.
    """
    beat_length = 60.0 / bpm * frames_per_second  # in frames
    delays = beat_length * np.arange(1, beats + 1)
    delays = delays[delays < len(repeat) - 1]
    if delays.size == 0:
        return 0.0
    return float(np.interp(delays, np.arange(len(repeat)), repeat).mean())


def detect_tempo(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Return (bpm, alternative bpm).

    No algorithm can always tell 70 BPM from 140 BPM, because both fit
    the same rhythm. So we answer inside 70-180 and also return the
    half / double reading for the person to choose.
    """
    # onset strength: one number per frame, high where a new sound starts.
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    frames_per_second = sr / HOP_LENGTH

    # Autocorrelation: compare the rhythm with a delayed copy of itself,
    # for every delay up to 12 seconds.
    repeat = librosa.autocorrelate(onset_strength, max_size=int(12 * frames_per_second))
    repeat = repeat / (repeat[0] + 1e-9)

    # Step 1, rough search. A good tempo repeats on beats AND on whole
    # bars (bpm / 4 means "one bar of 4 beats"). Checking bars stops
    # syncopated kicks from fooling us into 2/3 of the real tempo.
    candidates = np.arange(MIN_BPM, MAX_BPM, 0.25)
    scores = [
        _tempo_score(repeat, frames_per_second, bpm, beats=8)
        + _tempo_score(repeat, frames_per_second, bpm / 4, beats=4)
        for bpm in candidates
    ]
    rough_bpm = candidates[int(np.argmax(scores))]

    # Step 2, fine search around the rough answer, using every beat in
    # the 12 seconds. More beats make a small tempo error easier to see.
    fine_candidates = np.arange(rough_bpm - 1.5, rough_bpm + 1.5, 0.05)
    fine_scores = [
        _tempo_score(repeat, frames_per_second, bpm, beats=1000)
        for bpm in fine_candidates
    ]
    bpm = float(fine_candidates[int(np.argmax(fine_scores))])

    alt_bpm = bpm * 2 if bpm < 100 else bpm / 2
    return round(bpm, 1), round(alt_bpm, 1)


# ------------------------------------------------------------------ key


def relative_key(key: str) -> str:
    """Return the key that uses the same notes: C major <-> A minor."""
    if key == NO_KEY:
        return NO_KEY
    tonic, mode = key.split(" ")
    index = NOTE_NAMES.index(tonic)
    if mode == "major":
        return f"{NOTE_NAMES[(index - 3) % 12]} minor"
    return f"{NOTE_NAMES[(index + 3) % 12]} major"


def detect_key(y: np.ndarray, sr: int) -> tuple[str, float, str]:
    """Return (best key, its confidence, second-best key).

    Idea: measure how much energy each of the 12 notes has across the
    track, then compare that picture with the typical picture of all
    24 keys and pick the closest match.
    """
    # Drums have no pitch and only blur the picture, so remove them first.
    harmonic = librosa.effects.harmonic(y)

    # chroma: 12 rows (C, C#, ... B), one column per moment in time.
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    note_energy = chroma.mean(axis=1)

    # A drum loop has no notes at all: every row is equally empty, so
    # there is no key to find.
    if not np.all(np.isfinite(note_energy)) or note_energy.std() < 1e-6:
        return NO_KEY, 0.0, NO_KEY

    scores = []
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            # np.roll shifts the profile so its tonic lands on this note.
            shifted_profile = np.roll(profile, tonic)
            score = float(np.corrcoef(note_energy, shifted_profile)[0, 1])
            scores.append((score, f"{NOTE_NAMES[tonic]} {mode}"))

    scores.sort(reverse=True)
    best_score, best_key = scores[0]
    second_key = scores[1][1]

    # A weak match means the notes do not really fit any key (for example
    # a kick drum's pitch), so it is more honest to give no answer.
    if best_score < MIN_KEY_CONFIDENCE:
        return NO_KEY, round(best_score, 2), NO_KEY
    return best_key, round(best_score, 2), second_key


# ----------------------------------------------------------- everything


def analyze(path: str) -> AnalysisResult:
    """Load an audio file and return its tempo and key."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS)

    if y.size == 0 or float(np.abs(y).max()) < 1e-4:
        raise ValueError("The file is empty or silent, there is nothing to analyse.")

    bpm, alt_bpm = detect_tempo(y, sr)
    key, confidence, second_key = detect_key(y, sr)

    return AnalysisResult(
        bpm=bpm,
        alt_bpm=alt_bpm,
        key=key,
        relative_key=relative_key(key),
        key_confidence=confidence,
        second_key=second_key,
        duration_seconds=round(float(librosa.get_duration(path=path)), 1),
    )
