"""BeatScan web page: drop an audio file, get its key and BPM.

Run it with:  streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from analysis import NO_KEY, AnalysisResult, analyze

AUDIO_TYPES = ["wav", "mp3", "flac", "ogg"]

st.set_page_config(page_title="BeatScan", page_icon="🎧", layout="centered")


@st.cache_data(show_spinner=False)
def analyze_bytes(file_bytes: bytes, extension: str) -> AnalysisResult:
    """Save the upload to a temporary file and analyse it.

    st.cache_data remembers the answer, so the same file is not analysed
    twice when the page redraws.
    """
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"upload{extension}"
        path.write_bytes(file_bytes)
        return analyze(str(path))


def format_duration(seconds: float) -> str:
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}:{rest:02d}"


st.title("🎧 BeatScan")
st.caption("Drop a track to find its key and BPM.")

uploaded = st.file_uploader("Audio file", type=AUDIO_TYPES)

if uploaded is None:
    st.info("Drag a file onto the box above, or click it to browse.")
    st.stop()

st.audio(uploaded)

try:
    with st.spinner("Analysing..."):
        result = analyze_bytes(uploaded.getvalue(), Path(uploaded.name).suffix.lower())
except Exception as error:  # show the problem instead of a crash page
    st.error(f"Could not analyse this file: {error}")
    st.stop()

key_column, bpm_column, length_column = st.columns(3)
key_column.metric("Key", result.key)
bpm_column.metric("BPM", f"{result.bpm:g}")
length_column.metric("Length", format_duration(result.duration_seconds))

if result.key == NO_KEY:
    st.caption("The notes do not fit any key well. This sounds like drums only.")
else:
    st.caption(
        f"{result.key} uses the same notes as {result.relative_key}, "
        "so the two are easy to confuse. "
        f"Confidence {result.key_confidence:.2f}, second guess: {result.second_key}."
    )

st.caption(
    f"If the tempo feels wrong, try {result.alt_bpm:g} BPM: "
    "half-time and double-time fit the same rhythm."
)
