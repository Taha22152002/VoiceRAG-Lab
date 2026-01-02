import os
import logging
from datetime import datetime, timedelta
import re
import tempfile
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS
import requests
from elevenlabs.client import ElevenLabs
import logging
logging.basicConfig(level=logging.DEBUG)


# Import core components and utilities
from rag_core import RagBot 
from utils import setup_cors, parse_grounding_metadata

# Import the new Blueprint containing /ingest/* routes
from ingestion_routes import ingestion_bp
from appointment_routes import appointment_bp
import threading

# --- Initialization ---
load_dotenv()


def create_app(api_key):
    """Factory function to create and configure the Flask application."""
    
    app = Flask(__name__)
    setup_cors(app) # Use utility function for CORS
    
    # Initialize the RAG core logic
    rag_system = RagBot(api_key=api_key)
    
    # Store RagBot instance in config for access by Blueprints/Routes
    app.config['RAG_SYSTEM'] = rag_system
    app.config['EMBEDDING_MODEL'] = rag_system.EMBEDDING_MODEL
    app.config['CHAT_MODEL'] = rag_system.CHAT_MODEL

    # ======================================================
    # 🔌 BLUEPRINT REGISTRATION (FIX)
    # ======================================================
    # Register the ingestion blueprint and apply the '/ingest' URL prefix here.
    # Note: If you already had 'url_prefix="/ingest"' in ingestion_routes.py, 
    # you would remove the prefix argument here to avoid double-prefixing.
    app.register_blueprint(ingestion_bp, url_prefix='/ingest')
    app.register_blueprint(appointment_bp)


    # Optional: handle OPTIONS preflight globally
    @app.route("/", methods=["OPTIONS"])
    def options_root():
        return make_response("", 204)

    # ======================================================
    # 💬 CHAT ENDPOINT (Remains in app.py as a core function)
    # ======================================================
    @app.route("/chat", methods=["POST"])
    def chat_with_rag():
        """Handles chat requests and optionally retrieves RAG context."""
        # Retrieve rag_system from the app config
        rag_system = app.config['RAG_SYSTEM'] 
        
        data = request.json
        user_message = data.get("userMessage")
        system_prompt = data.get("systemPrompt", "")
        chat_history = data.get("history", [])

        if not user_message:
            return jsonify({"error": "Missing userMessage."}), 400

        # --- Relative date normalization (today/tomorrow/day after tomorrow) ---
        def normalize_relative_dates(text: str) -> str:
            if not isinstance(text, str) or not text:
                return text
            today = datetime.now().date()
            mapping = {
                r"\btoday\b": today.isoformat(),
                r"\btomorrow\b": (today + timedelta(days=1)).isoformat(),
                r"\bday\s+after\s+tomorrow\b": (today + timedelta(days=2)).isoformat(),
            }
            normalized = text
            # Replace longer phrases first to avoid partial matches
            for pattern, replacement in [(r"\bday\s+after\s+tomorrow\b", mapping[r"\bday\s+after\s+tomorrow\b"]),
                                         (r"\btomorrow\b", mapping[r"\btomorrow\b"]),
                                         (r"\btoday\b", mapping[r"\btoday\b"])]:
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
            return normalized

        normalized_message = normalize_relative_dates(user_message)

        try:
            # Check if user wants to book appointments (enable tool calling)
            booking_keywords = ['book', 'appointment', 'schedule', 'slot', 'service', 'wash']

            def contains_iso_date(text: str) -> bool:
                try:
                    return bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b", text))
                except Exception:
                    return False

            def has_booking_context(history: list) -> bool:
                try:
                    context_keywords = ['slot', 'slots', 'appointment', 'book', 'schedule', 'service', 'wash']
                    recent = history[-3:] if isinstance(history, list) else []
                    for msg in recent:
                        text = (msg or {}).get('text', '')
                        role = (msg or {}).get('role', '')
                        if role == 'model' and any(k in text.lower() for k in context_keywords):
                            return True
                    return False
                except Exception:
                    return False

            lower_msg = normalized_message.lower()
            def contains_relative_date(text: str) -> bool:
                try:
                    return any(p in text for p in ['today', 'tomorrow', 'day after tomorrow'])
                except Exception:
                    return False

            def history_contains_date(history: list) -> bool:
                try:
                    recent = history[-5:] if isinstance(history, list) else []
                    for msg in recent:
                        t = (msg or {}).get('text', '').lower()
                        if contains_iso_date(t) or contains_relative_date(t):
                            return True
                    return False
                except Exception:
                    return False

            booking_intent = any(keyword in lower_msg for keyword in booking_keywords) or has_booking_context(chat_history)
            has_date = contains_iso_date(lower_msg) or contains_relative_date(lower_msg) or history_contains_date(chat_history)
            enable_tools = booking_intent and has_date
            
            if enable_tools:
                # Use tool calling for booking-related queries
                result = rag_system.generate_response_with_tools(
                    user_message=normalized_message,
                    system_prompt=system_prompt,
                    chat_history=chat_history,
                    user_id=data.get("user_id", "guest")
                )
                
                return jsonify({
                    "response": result['response'],
                    "tool_used": result['tool_used'],
                    "tool_result": result['tool_result'],
                    "mode": result['mode']
                }), 200
            else:
                # Regular RAG response
                response = rag_system.generate_response(normalized_message, system_prompt, chat_history)
                # Safely extract text
                model_response = getattr(response, "text", None)
                if not model_response:
                    try:
                        if getattr(response, "candidates", None):
                            candidate = response.candidates[0]
                            parts = getattr(candidate, "content", None)
                            if parts and getattr(parts, "parts", None):
                                maybe_part = parts.parts[0]
                                model_response = getattr(maybe_part, "text", None)
                    except Exception:
                        model_response = None

                if not model_response:
                    model_response = "I'm sorry, I couldn't generate a response."
                
                # Use utility for source parsing
                sources = parse_grounding_metadata(response)
                
                return jsonify({
                    "response": model_response,
                    "sources": sources,
                    "tool_used": None,
                    "tool_result": None,
                    "mode": "RAG" if rag_system.vector_store else "Base LLM"
                }), 200

        except Exception as e:
            print(f"Gemini API Error: {e}")
            return jsonify({"error": f"LLM Generation failed: {str(e)}"}), 500

    # =============================
    # 🎤 VOICE STT ENDPOINT
    # =============================

    def stt_result_to_dict(stt_result):
        """Convert ElevenLabs STT SDK object to JSON-serializable dict"""
        result = {
            'text': getattr(stt_result, 'text', ''),
            'language_code': getattr(stt_result, 'language_code', None),
            'language_probability': getattr(stt_result, 'language_probability', None),
            'transcription_id': getattr(stt_result, 'transcription_id', None),
            'words': []
        }

        words = getattr(stt_result, 'words', [])
        try:
            for w in words:
                result['words'].append({
                    'text': getattr(w, 'text', ''),
                    'start': getattr(w, 'start', None),
                    'end': getattr(w, 'end', None),
                    'type': getattr(w, 'type', None),
                    'speaker_id': getattr(w, 'speaker_id', None),
                    'logprob': getattr(w, 'logprob', None),
                })
        except Exception:
            # If words is not iterable or SDK changed type
            pass

        return result
    @app.route('/voice/stt', methods=['POST'])
    def voice_stt():
        """Handles speech-to-text conversion for voice input using ElevenLabs STT (Scribe v1/v2)."""
        try:
            # 1️⃣ Check for audio file in request
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400

            audio_file = request.files['audio']

            # 2️⃣ Read audio bytes into memory
            from io import BytesIO
            try:
                audio_bytes = audio_file.read()
            except Exception:
                audio_bytes = audio_file.stream.read()
            if not audio_bytes or len(audio_bytes) < 256:
                return jsonify({'error': 'Audio payload too small or empty'}), 400

            audio_data = BytesIO(audio_bytes)
            audio_data.seek(0)

            # 3️⃣ Initialize ElevenLabs client
            ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
            if not ELEVENLABS_API_KEY:
                return jsonify({'error': 'Missing ELEVENLABS_API_KEY in environment'}), 500

            client = ElevenLabs(api_key=ELEVENLABS_API_KEY, base_url='https://api.elevenlabs.io')

            # 4️⃣ Parse optional parameters
            language_code = request.form.get('language_code') or 'eng'
            diarize = str(request.form.get('diarize', 'true')).lower() in {'1', 'true', 't', 'yes', 'y'}

            # 5️⃣ Perform STT using SDK
            stt_result = client.speech_to_text.convert(
                file=audio_data,
                model_id='scribe_v1',  # or 'scribe_v2' if you prefer
                tag_audio_events=True,
                language_code=language_code,
                diarize=diarize,
            )

            # 6️⃣ Extract transcribed text
            transcribed_text = getattr(stt_result, 'text', None)
            if not transcribed_text:
                return jsonify({'error': 'Transcription empty', 'raw': stt_result_to_dict(stt_result)}), 502

            # 7️⃣ Return result
            return jsonify({'text': transcribed_text, 'raw': stt_result_to_dict(stt_result)}), 200

        except Exception as e:
            logging.error("Voice STT Exception: %s", traceback.format_exc())
            return jsonify({'error': 'Failed to process voice message','details': str(e)
                            }), 500

    # =============================
    # 🔊 VOICE TTS ENDPOINT
    # =============================
    @app.route('/voice/tts', methods=['POST'])
    def voice_tts():
        """Converts text to speech using ElevenLabs TTS and returns audio as HTTP response."""
        try:
            # Parse incoming JSON
            data = request.get_json(force=True, silent=True) or {}
            text = data.get('text')
            voice_id = data.get('voice_id', '0QT4OrDTvpDlUPmFsUWN')
            output_format = data.get('output_format', 'mp3_44100_128')
            model_id = data.get('model_id', 'eleven_multilingual_v2')

            # Validate text input
            if not text or not isinstance(text, str):
                return jsonify({'error': 'Missing or invalid "text" in request body'}), 400

            # Load API key from environment
            api_key = os.getenv('ELEVENLABS_API_KEY')
            if not api_key:
                return jsonify({'error': 'Missing ELEVENLABS_API_KEY in environment'}), 500

            # Initialize ElevenLabs client (fix base_url)
            client = ElevenLabs(api_key=api_key, base_url='https://api.elevenlabs.io')

            # Generate audio
            try:
                audio_result = client.text_to_speech.convert(
                    voice_id=voice_id,
                    output_format=output_format,
                    text=text,
                    model_id=model_id
                )

                # Handle both generator and bytes return types
                if hasattr(audio_result, '__iter__') and not isinstance(audio_result, (bytes, bytearray)):
                    audio_bytes = b"".join(audio_result)
                else:
                    audio_bytes = audio_result

            except Exception as e:
                return jsonify({'error': 'ElevenLabs TTS failed', 'details': str(e)}), 502

            # Prepare in-memory file for response
            from io import BytesIO
            audio_io = BytesIO(audio_bytes)
            audio_io.seek(0)

            # Determine MIME type
            mimetype = 'audio/mpeg' if output_format.startswith('mp3') else 'audio/wav'

            # Send audio as HTTP response (inline playback)
            return send_file(
                audio_io,
                mimetype=mimetype,
                as_attachment=False,
                download_name='speech.' + ('mp3' if output_format.startswith('mp3') else 'wav')
            )

        except Exception as e:
            print(f"[voice_tts] Unexpected error: {str(e)}")
            return jsonify({'error': 'Failed to generate speech', 'details': str(e)}), 500

    return app


# ======================================================
# 🚀 APP ENTRY POINT
# ======================================================
if __name__ == "__main__":
    # --- Simplified Google API Setup ---
    API_KEY = os.getenv("GOOGLE_API_KEY")

    if not API_KEY:
        raise ValueError("❌ Missing GOOGLE_API_KEY in environment.")
    os.environ["GOOGLE_API_KEY"] = API_KEY
    
    # Create the app instance using the factory
    app = create_app(API_KEY)

    # Note on command-line error: The correct way to specify port is 'PORT=5050 python app.py'
    port = int(os.getenv("PORT", "5200")) 

    def run_ws():
        from ws_server import start_ws_server
        # WebSocket server listens on 5201 to avoid conflict with HTTP 5200
        start_ws_server(app.config['RAG_SYSTEM'], host="127.0.0.1", port=5201)

    t = threading.Thread(target=run_ws, daemon=True)
    t.start()

    app.run(debug=True, host="127.0.0.1", port=port)