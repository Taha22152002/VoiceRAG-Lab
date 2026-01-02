import asyncio
import json
import re
from datetime import datetime, timedelta
import websockets

async def ws_handler(websocket, path, rag_system):
    session = {"system_prompt": "", "history": []}

    async for message in websocket:
        try:
            data = json.loads(message)
        except Exception:
            await websocket.send(json.dumps({"type": "error", "message": "invalid_json"}))
            continue

        msg_type = data.get("type")
        if msg_type == "session_start":
            session["system_prompt"] = data.get("systemPrompt", "")
            await websocket.send(json.dumps({"type": "session_ack"}))
            continue

        if msg_type == "user_message":
            user_message = data.get("text")
            message_id = data.get("messageId")
            history = data.get("history") or session["history"]

            if not user_message:
                await websocket.send(json.dumps({"type": "error", "message": "missing_user_message", "messageId": message_id}))
                continue

            def normalize_relative_dates(text):
                if not isinstance(text, str) or not text:
                    return text
                today = datetime.now().date()
                mapping = {
                    r"\btoday\b": today.isoformat(),
                    r"\btomorrow\b": (today + timedelta(days=1)).isoformat(),
                    r"\bday\s+after\s+tomorrow\b": (today + timedelta(days=2)).isoformat(),
                }
                normalized = text
                for pattern, replacement in [(r"\bday\s+after\s+tomorrow\b", mapping[r"\bday\s+after\s+tomorrow\b"]),
                                             (r"\btomorrow\b", mapping[r"\btomorrow\b"]),
                                             (r"\btoday\b", mapping[r"\btoday\b"])]:
                    normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
                return normalized

            normalized_message = normalize_relative_dates(user_message)

            try:
                booking_keywords = ['book', 'appointment', 'schedule', 'slot', 'service', 'wash']

                def contains_iso_date(text):
                    try:
                        return bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b", text))
                    except Exception:
                        return False

                def has_booking_context(history_list):
                    try:
                        context_keywords = ['slot', 'slots', 'appointment', 'book', 'schedule', 'service', 'wash']
                        recent = history_list[-3:] if isinstance(history_list, list) else []
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

                def history_contains_date(history_list):
                    try:
                        recent = history_list[-5:] if isinstance(history_list, list) else []
                        for msg in recent:
                            t = (msg or {}).get('text', '').lower()
                            if contains_iso_date(t) or contains_relative_date(t):
                                return True
                        return False
                    except Exception:
                        return False

                booking_intent = any(keyword in lower_msg for keyword in booking_keywords) or has_booking_context(history)
                has_date = contains_iso_date(lower_msg) or contains_relative_date(lower_msg) or history_contains_date(history)
                enable_tools = booking_intent and has_date

                if enable_tools:
                    result = rag_system.generate_response_with_tools(
                        user_message=normalized_message,
                        system_prompt=session["system_prompt"],
                        chat_history=history,
                        user_id=data.get("user_id", "guest")
                    )
                    session["history"].append({"role": "user", "text": normalized_message})
                    session["history"].append({"role": "model", "text": result.get('response', '')})
                    await websocket.send(json.dumps({
                        "type": "model_response_done",
                        "messageId": message_id,
                        "response": result.get('response', ''),
                        "tool_used": result.get('tool_used'),
                        "tool_result": result.get('tool_result'),
                        "mode": result.get('mode')
                    }))
                else:
                    # Stream model response in chunks
                    accumulated = []
                    try:
                        for delta in rag_system.generate_response_stream(normalized_message, session["system_prompt"], history):
                            if isinstance(delta, str) and delta:
                                accumulated.append(delta)
                                await websocket.send(json.dumps({"type": "model_response_chunk", "messageId": message_id, "delta": delta}))
                    except Exception as e:
                        await websocket.send(json.dumps({"type": "error", "message": str(e), "messageId": message_id}))
                        continue
                    full_text = "".join(accumulated) if accumulated else "I'm sorry, I couldn't generate a response."
                    session["history"].append({"role": "user", "text": normalized_message})
                    session["history"].append({"role": "model", "text": full_text})
                    await websocket.send(json.dumps({
                        "type": "model_response_done",
                        "messageId": message_id,
                        "response": full_text,
                        "mode": "RAG" if rag_system.vector_store else "Base LLM"
                    }))
            except Exception as e:
                await websocket.send(json.dumps({"type": "error", "message": str(e), "messageId": message_id}))

def start_ws_server(rag_system, host="127.0.0.1", port=5201):
    async def main():
        async def handler(websocket, path):
            await ws_handler(websocket, path, rag_system)
        await websockets.serve(handler, host, port, ping_interval=20, ping_timeout=20)
        await asyncio.Future()
    asyncio.run(main())