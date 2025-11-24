#!/usr/bin/env python3
"""Transcribe audio from a URL using Google GenAI (Gemini).

Usage:
  python transcribe_from_link.py <url> [--output out.txt]

Requirements:
  - See `requirements.txt` and `setup.sh`.
  - System `ffmpeg` must be installed for `yt-dlp` audio extraction.

This script:
  - Downloads audio from the provided URL (YouTube and other sites supported by yt-dlp).
  - Uploads the audio file to the Google GenAI Files API.
  - Sends a Jinja2-rendered prompt and the uploaded audio file to the Gemini model.
  - Post-processes and writes a cleaned transcript to disk.
"""
import argparse
import logging
import os
import sys
import tempfile
import glob

from dotenv import load_dotenv
from jinja2 import Template

try:
    # Defer heavy/optional imports to functions so unit tests can import this module
    genai = None
except Exception:
    genai = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


PROMPT_TEMPLATE = Template("""Generate a transcript of the episode. Include timestamps and identify speakers.

Speakers are:
{% for speaker in speakers %}- {{ speaker }}{% if not loop.last %}\n{% endif %}{% endfor %}

eg:
[00:00] Brady: Hello there.
[00:02] Tim: Hi Brady.

Instructions:
- Use short captions (one or two short sentences) with a leading timestamp in HH:MM:SS format.
- Identify speakers with the exact names above. If unknown, use single letters (A, B, ...).
- Mark music/jingles or other sounds in square brackets, e.g. [MUSIC] or [Bell ringing]. If you can identify the song, include the title.
- Use only English alphabet characters unless foreign characters are correct.
- Spell names of people, movies, books or places correctly — use context.
- Do NOT use markdown formatting; output plain text only.
- End the transcript with the tag [END].

Produce the transcript as plain text with one caption per line.
""")


def download_audio(url: str, target_dir: str) -> str:
    """Download audio from URL into target_dir and return local filepath."""
    import yt_dlp

    # For direct mp3 links we can skip ffmpeg postprocessing (avoids requiring system ffmpeg)
    needs_conversion = not url.lower().split('?')[0].endswith('.mp3')
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(target_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    if needs_conversion:
        # Use ffmpeg to convert to mp3 for consistent upload when needed
        ydl_opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]

    logging.info("Downloading audio from %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # Construct expected filename
    ext = "mp3"
    filename = f"{info.get('id')}.{ext}"
    filepath = os.path.join(target_dir, filename)
    if not os.path.exists(filepath):
        # try to find any file in target_dir
        files = glob.glob(os.path.join(target_dir, "*.*"))
        if files:
            filepath = files[0]
        else:
            raise FileNotFoundError("Downloaded audio file not found")

    logging.info("Saved audio to %s", filepath)
    return filepath


def timestamp_to_seconds(ts_str):
    try:
        ts_str = ts_str.split(".")[0]
        parts = list(map(int, ts_str.split(':')))
        if len(parts) == 3:
            h, m, s = parts
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = parts
            return m * 60 + s
    except Exception:
        return None


def seconds_to_timestamp(total_seconds):
    if total_seconds is None or total_seconds < 0:
        total_seconds = 0
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def process_transcript(input_text, max_segment_duration=30):
    import re

    lines = input_text.strip().splitlines()
    output_lines = []

    current_segment_start_ts_str = None
    current_segment_start_seconds = None
    current_speaker = None
    current_text_parts = []

    line_regex = re.compile(r'^\[((?:\d{2}:)?\d{2}:\d{2}(?:\.\d+)?)\]\s*([^:]+?):\s*(.*)$')

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        match = line_regex.match(line)
        if not match:
            if current_speaker is not None:
                segment_text = ' '.join(filter(None, current_text_parts))
                output_lines.append(f"[{current_segment_start_ts_str}] {current_speaker}: {segment_text}")
                current_speaker = None
                current_text_parts = []
                current_segment_start_ts_str = None
                current_segment_start_seconds = None
            output_lines.append(line)
            continue

        ts_str, speaker, text = match.groups()
        speaker = speaker.strip()
        text = text.strip()
        current_seconds = timestamp_to_seconds(ts_str)
        if current_seconds is None:
            logging.warning("Skipping line %d due to invalid timestamp: %s", i + 1, line)
            continue

        start_new_segment = False
        if current_speaker is None:
            start_new_segment = True
        elif speaker != current_speaker:
            start_new_segment = True
        elif current_segment_start_seconds is not None and current_seconds - current_segment_start_seconds > max_segment_duration:
            start_new_segment = True

        if start_new_segment:
            if current_speaker is not None:
                segment_text = ' '.join(filter(None, current_text_parts))
                output_lines.append(f"[{current_segment_start_ts_str}] {current_speaker}: {segment_text}")

            current_segment_start_ts_str = seconds_to_timestamp(current_seconds)
            current_segment_start_seconds = current_seconds
            current_speaker = speaker
            current_text_parts = [text]
        else:
            if text:
                current_text_parts.append(text)

    if current_speaker is not None:
        segment_text = ' '.join(filter(None, current_text_parts))
        output_lines.append(f"[{current_segment_start_ts_str}] {current_speaker}: {segment_text}")

    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(description="Download audio from a link and transcribe using Google Gemini.")
    parser.add_argument("input", help="URL or local path to audio file (URL if starts with http)")
    parser.add_argument("--speakers", help="Comma-separated speaker names (optional)", default="")
    parser.add_argument("--output", help="Output transcript file", default="transcript.txt")
    parser.add_argument("--dry-run", help="Only download (or validate) the file and exit", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    # API key optional if user only wants a dry-run or is passing a local file
    if not api_key:
        logging.warning("GOOGLE_API_KEY not set. API calls will be skipped unless you provide --dry-run.")

    # Lazy import of google-genai to avoid hard dependency during tests
    client = None
    if api_key and not args.dry_run:
        try:
            from google import genai as _genai
            client = _genai.Client(api_key=api_key)
        except Exception as e:
            logging.error("Failed to import or construct google-genai client: %s", e)
            client = None

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    if not speakers:
        speakers = ["Host", "Guest"]

    prompt = PROMPT_TEMPLATE.render(speakers=speakers)

    # Determine if input is URL or local file
    is_url = args.input.lower().startswith("http")

    if args.dry_run:
        # For dry-run: validate or download file and exit
        if is_url:
            with tempfile.TemporaryDirectory() as tmpdir:
                audio_path = download_audio(args.input, tmpdir)
                logging.info("Dry-run: downloaded to %s", audio_path)
        else:
            if os.path.exists(args.input):
                logging.info("Dry-run: local file exists: %s", args.input)
            else:
                logging.error("Dry-run: local file not found: %s", args.input)
                sys.exit(1)
        logging.info("Dry-run complete. Exiting.")
        return

    # Normal flow: either use local file or download first
    if is_url:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio(args.input, tmpdir)
            uploaded_file = None
            if client:
                logging.info("Uploading audio to Files API...")
                uploaded_file = client.files.upload(file=audio_path)
            else:
                logging.error("No GenAI client available. Set GOOGLE_API_KEY to enable transcription.")
                sys.exit(1)

            logging.info("Requesting transcription from Gemini model...")
            response = client.models.generate_content(
                model="gemini-2.5-pro-exp-03-25",
                contents=[prompt, uploaded_file],
            )

    else:
        # local file
        audio_path = args.input
        if not os.path.exists(audio_path):
            logging.error("Local file not found: %s", audio_path)
            sys.exit(1)
        if not client:
            logging.error("No GenAI client available. Set GOOGLE_API_KEY to enable transcription.")
            sys.exit(1)
        logging.info("Uploading local audio to Files API...")
        uploaded_file = client.files.upload(file=audio_path)
        logging.info("Requesting transcription from Gemini model...")
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=[prompt, uploaded_file],
        )

    raw_text = getattr(response, "text", str(response))
    logging.info("Raw model output length: %d", len(raw_text))

    processed = process_transcript(raw_text, max_segment_duration=30)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(processed)

    logging.info("Transcript written to %s", args.output)


if __name__ == "__main__":
    main()
