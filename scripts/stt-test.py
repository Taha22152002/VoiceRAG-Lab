# example_converted.py
import os
from dotenv import load_dotenv
import requests # Still needed if you might switch back, but not for local file
from elevenlabs.client import ElevenLabs

# --- Configuration ---
load_dotenv()

elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

# --- Local File Path ---
# Change the path to your local file
local_file_path = "data/test.mp3"
# The requests and BytesIO imports are no longer strictly needed for a local file,
# but keeping them doesn't hurt if you plan to re-use the code later.

# --- Transcribe Local File ---
# Open the file in binary read mode ('rb')
with open(local_file_path, "rb") as audio_file:
    transcription = elevenlabs.speech_to_text.convert(
        # Pass the file object directly
        file=audio_file,
        model_id="scribe_v2", # Model to use
        tag_audio_events=True, # Tag audio events like laughter, applause, etc.
        language_code="eng", # Language of the audio file. If set to None, the model will detect the language automatically.
        diarize=True, # Whether to annotate who is speaking
    )

print(transcription)