from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import numpy as np

MODULE = Path(__file__).parents[1] / "main.py"
SPEC = importlib.util.spec_from_file_location("synapse_whisper_xdna_main", MODULE)
assert SPEC and SPEC.loader
main = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(main)


def test_merge_text_removes_word_overlap() -> None:
    assert main.merge_text("one two three", "two three four") == "one two three four"


def test_timestamp_stitch_assigns_overlap_once() -> None:
    segments = [{"start": 0.0, "end": 10.0, "text": "left"}, {"start": 10.0, "end": 20.0, "text": "old"}]
    incoming = [{"start": 0.0, "end": 5.0, "text": "new"}, {"start": 5.0, "end": 15.0, "text": "right"}]
    main.append_segments(segments, incoming, base_time=10.0, previous_end=20.0)
    assert [item["text"] for item in segments] == ["left", "old", "right"]


def test_final_chunk_is_anchored_to_audio_end(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    samples = np.zeros(60 * 16000, dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(samples.tobytes())
    chunks = main.load_chunks(path, 30.0, 1.0)
    assert [offset for offset, _ in chunks] == [0.0, 29.0, 30.0]
    assert all(len(chunk) == 30 * 16000 for _, chunk in chunks)
