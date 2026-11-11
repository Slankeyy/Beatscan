# BeatScan

Drop an audio file, get its musical key and tempo (BPM).

This is the first stage of a tool for beatmakers. Planned next: chord
progression detection, MIDI export, and a searchable library of analysed
tracks.

## What it does now

- Accepts WAV, MP3, FLAC and OGG files by drag and drop
- Detects the tempo and also shows the half-time / double-time reading
- Detects the key, with a confidence score and a second guess
- Plays the uploaded file in the page

## How it works

- **Tempo**: `librosa` measures where new sounds start. The rhythm is then
  compared with delayed copies of itself (autocorrelation), and the tempo
  whose beats and 4-beat bars line up best with those repeats wins. This
  handles syncopated kicks better than a plain beat tracker, which often
  answers 2/3 of the real tempo on trap-style drums.
- **Key**: the drums are removed, the energy of each of the 12 notes is
  measured across the track, and that picture is compared with the
  Krumhansl-Kessler profiles of all 24 major and minor keys.

Both are estimates. Relative keys (for example C major and A minor) and
half / double tempo are the usual mistakes, which is why the page shows
the alternatives. Accuracy on real tracks has not been measured yet; that
is the next step.

## Run it

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux the second line is `source .venv/bin/activate`.

## Tests

```
pytest
```

## Project layout

```
analysis.py             tempo and key detection (no interface code)
app.py                  the Streamlit page
tests/test_analysis.py  tests on generated audio with known answers
```
