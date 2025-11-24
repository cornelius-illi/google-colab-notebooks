import os
import tempfile

from transcribe_from_link import (
    timestamp_to_seconds,
    seconds_to_timestamp,
    process_transcript,
)


def test_timestamp_to_seconds_and_back():
    assert timestamp_to_seconds("00:00:00") == 0
    assert timestamp_to_seconds("00:01:30") == 90
    assert timestamp_to_seconds("1:02:03") == 3723 or timestamp_to_seconds("01:02:03") == 3723
    assert seconds_to_timestamp(90) == "00:01:30"
    assert seconds_to_timestamp(None) == "00:00:00"


def test_process_transcript_merge_same_speaker():
    text = """
[00:00] Host: Hello and welcome.
[00:10] Host: Today we talk about testing.
[00:50] Guest: Thanks for having me.
"""
    processed = process_transcript(text, max_segment_duration=30)
    # Host lines should be merged (00:00 -> 00:10 within 30s)
    assert "[00:00:00] Host: Hello and welcome. Today we talk about testing." in processed
    assert "[00:00:50] Guest: Thanks for having me." in processed


def test_process_transcript_handles_nonmatch_lines():
    text = """
[00:00] Host: Intro.
[00:05] [MUSIC]
[00:10] Host: Back.
"""
    processed = process_transcript(text, max_segment_duration=30)
    assert "[00:00:00] Host: Intro." in processed
    assert "[00:00:05] [MUSIC]" in processed or "[MUSIC]" in processed
