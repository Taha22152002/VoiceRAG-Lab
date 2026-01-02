# example.py - Corrected Code

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from elevenlabs.client import ElevenLabs

# Note: The API key below is a placeholder/example key from your previous input.
# For production, it's best to use os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)


# Generate speech (MP3)
# The 'convert' method returns a generator/iterator of audio chunks (bytes).
audio_stream = client.text_to_speech.convert(
    text="The first move is what sets everything in motion.",
    voice_id="4cYYCJZtDFOV61Rs1Eoa",  # Example voice ID
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

# FIX: Collect all audio chunks from the generator into a single bytes object
# b"" starts an empty bytes object, and .join() concatenates all chunks.
audio_bytes = b"".join(audio_stream)

# Save MP3 file
mp3_file = "speech.mp3"
with open(mp3_file, "wb") as f:
    # Write the complete bytes object to the file
    f.write(audio_bytes)

print(f"✅ Audio saved successfully to {mp3_file}")


# --- Optional Audio Playback (Requires additional libraries like pydub and ffmpeg) ---
# To play the audio, you would typically use a library like pydub:
# from pydub import AudioSegment
# from pydub.playback import play
# try:
#     song = AudioSegment.from_mp3(mp3_file)
#     play(song)
# except Exception as e:
#     print(f"Warning: Could not play audio. Ensure 'pydub' is installed and 'ffmpeg' is accessible. Details: {e}")