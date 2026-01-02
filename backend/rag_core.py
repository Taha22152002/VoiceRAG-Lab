import json
import requests
from flask import jsonify
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.genai import Client, types
from datetime import datetime, timedelta
from bs4 import BeautifulSoup 
from typing import Dict, List, Any

# Import booking tools
from booking_tools import BookingToolExecutor, BOOKING_TOOLS 

# --- Constants ---
SESSION_ID = "rag_session_123"
EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ======================================================
# 🧠 RAG Core Logic
# ======================================================

class RagBot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = Client(api_key=api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
        self.vector_store = None
        
        # Initialize booking tool executor
        # (Reference Implementation: Appointment Booking)
        self.booking_executor = BookingToolExecutor()
        # Remember last date used for slot lookup to infer booking date
        self.last_slots_date = None
        
        # Tool definitions for function calling
        # Replace this with your own list of tools for the framework
        self.tools = BOOKING_TOOLS
        
        # Expose constants for external use
        self.EMBEDDING_MODEL = EMBEDDING_MODEL
        self.CHAT_MODEL = CHAT_MODEL

    def store_documents(self, documents, source="unknown", count=0):
        """Helper to split, embed, and store any document list."""
        # Local import to prevent circular dependency if imported elsewhere
        from langchain_text_splitters import RecursiveCharacterTextSplitter 
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            # NEW: Specify keys to carry over from original Document metadata to all chunks
            # This is CRUCIAL for proper attribution and better retrieval performance
            keep_separator=True,
            length_function=len
        )
        # Note: The RecursiveCharacterTextSplitter automatically preserves metadata by default,
        # but explicitly passing a list of metadata to filter is usually a good practice 
        # when initializing the splitter in more complex setups. 
        # For the current setup, we will rely on the default behavior but keep the split robust.
        
        chunks = text_splitter.split_documents(documents)

        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings, 
                collection_name=SESSION_ID
            )
        else:
            self.vector_store.add_documents(chunks)

        return jsonify({
            "message": f"Successfully ingested {count} {source} entries ({len(chunks)} chunks). RAG ready."
        }), 200

    def get_rag_context(self, user_message):
        """Retrieves relevant context from the vector store."""
        if not self.vector_store:
            return None, ""

        try:
            # Setting k=4 is standard for a small knowledge base
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(user_message)
            
            # IMPROVEMENT: Add source metadata to the context for better grounding
            context_parts = []
            for i, doc in enumerate(docs):
                source_info = doc.metadata.get("url") or doc.metadata.get("name") or "Unknown Source"
                context_parts.append(f"--- Document Chunk {i+1} (Source: {source_info}) ---\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            return context, self.get_rag_prompt(user_message, context)
            
        except Exception as e:
            print(f"Retrieval failed: {e}")
            return None, user_message

    def fetch_link(self, link, name):
        """
        Fetches content from a URL, and now CLEANS the HTML to remove noise.
        """
        try:
            resp = requests.get(link, timeout=10) 
            # Ensure the response content is not empty before proceeding
            if resp.status_code == 200 and resp.text.strip(): 
                # --- NEW CLEANUP LOGIC ---
                soup = BeautifulSoup(resp.content, 'html.parser')
                # Remove script and style elements
                for script_or_style in soup(['script', 'style']):
                    script_or_style.decompose()
                
                # Get text and clean up whitespace
                clean_text = soup.get_text()
                lines = (line.strip() for line in clean_text.splitlines())
                # Use a single space separator for phrases to avoid excessive newlines
                chunks = (phrase.strip() for line in lines for phrase in line.split("  ")) 
                clean_content = ' '.join(chunk for chunk in chunks if chunk) # Joining with a space for better readability/chunking
                
                if not clean_content:
                    print(f"Failed to extract meaningful text from link: {link}")
                    return None
                    
                # Ensure metadata includes source for attribution
                return Document(page_content=clean_content, metadata={"name": name, "url": link, "source_type": "link"})
        except Exception as e:
            print(f"Failed to fetch link {link}: {e}")
            return None
        return None 

    def generate_response(self, user_message, system_prompt, chat_history):
        """Generates the final response using Gemini."""
        final_prompt = user_message
        tools = []
        gemini_history = []

        # 1. RAG Retrieval
        if self.vector_store:
            # The get_rag_context now returns a richer prompt with context
            context, final_prompt = self.get_rag_context(user_message) 

        # 2. History Conversion
        for msg in chat_history:
            role = 'user' if msg.get("role") == 'user' else 'model'
            text = msg.get("text")
            if text:
                # Ensure the message is not empty
                gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

        # Append the final prompt (RAG or original message)
        gemini_history.append(types.Content(role="user", parts=[types.Part.from_text(text=final_prompt)]))

        # 3. Tool Check
        if not self.vector_store:
            tools.append({"google_search": {}})

        # 4. Generation
        response = self.client.models.generate_content(
            model=CHAT_MODEL,
            contents=gemini_history,
            config=types.GenerateContentConfig(system_instruction=system_prompt, tools=tools)
        )
        # Return the raw response so callers can access .text safely
        return response

    def generate_response_stream(self, user_message, system_prompt, chat_history):
        final_prompt = user_message
        tools = []
        gemini_history = []

        if self.vector_store:
            context, final_prompt = self.get_rag_context(user_message)

        for msg in chat_history:
            role = 'user' if msg.get("role") == 'user' else 'model'
            text = msg.get("text")
            if text:
                gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

        gemini_history.append(types.Content(role="user", parts=[types.Part.from_text(text=final_prompt)]))
        if not self.vector_store:
            tools.append({"google_search": {}})

        accumulated = []
        for chunk in self.client.models.generate_content_stream(
            model=CHAT_MODEL,
            contents=gemini_history,
            config=types.GenerateContentConfig(system_instruction=system_prompt, tools=tools)
        ):
            try:
                if getattr(chunk, "text", None):
                    t = chunk.text
                    accumulated.append(t)
                    yield t
                else:
                    cand = getattr(chunk, "candidates", None)
                    if cand:
                        parts_obj = cand[0].content
                        parts = getattr(parts_obj, "parts", [])
                        for p in parts:
                            txt = getattr(p, "text", None)
                            if txt:
                                accumulated.append(txt)
                                yield txt
            except Exception:
                pass
        return "".join(accumulated)
    def execute_tool_call(self, function_call) -> Dict[str, Any]:
        """Execute a tool call and return the result"""
        function_name = function_call.name
        function_args = function_call.args

        print(f"Executing tool: {function_name} with args: {function_args}")

        # Normalize natural language dates like "today" / "tomorrow"
        def normalize_date(value):
            try:
                if isinstance(value, str):
                    v = value.strip().lower()
                    if v == "today":
                        return datetime.now().date().strftime("%Y-%m-%d")
                    if v == "tomorrow":
                        return (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
            except Exception:
                pass
            return value

        # Normalize time formats like "04:00pm", "4pm", "16:00" -> "4:00 PM"
        def normalize_time(value):
            try:
                if not isinstance(value, str):
                    return value
                v = value.strip().lower().replace(' ', '')
                mapping = {
                    '9:00am': '9:00 AM', '10:00am': '10:00 AM', '11:00am': '11:00 AM', '12:00pm': '12:00 PM',
                    '1:00pm': '1:00 PM', '2:00pm': '2:00 PM', '3:00pm': '3:00 PM', '4:00pm': '4:00 PM',
                    '9am': '9:00 AM', '10am': '10:00 AM', '11am': '11:00 AM', '12pm': '12:00 PM',
                    '1pm': '1:00 PM', '2pm': '2:00 PM', '3pm': '3:00 PM', '4pm': '4:00 PM',
                    '09:00': '9:00 AM', '10:00': '10:00 AM', '11:00': '11:00 AM', '12:00': '12:00 PM',
                    '13:00': '1:00 PM', '14:00': '2:00 PM', '15:00': '3:00 PM', '16:00': '4:00 PM'
                }
                # handle 04:00pm, 04:00 PM
                v = v.replace('pm', 'pm').replace('am', 'am')
                if v in mapping:
                    return mapping[v]
                # convert like 04:00pm -> 4:00 PM
                import re as _re
                m = _re.match(r'^(0?[1-9]|1[0-6]):00(pm|am)$', v)
                if m:
                    hh = m.group(1)
                    suffix = m.group(2).upper()
                    if suffix == 'AM':
                        if hh in ['09','9']: return '9:00 AM'
                        if hh in ['10']: return '10:00 AM'
                        if hh in ['11']: return '11:00 AM'
                    if suffix == 'PM':
                        if hh in ['12']: return '12:00 PM'
                        if hh in ['13','1']: return '1:00 PM'
                        if hh in ['14','2']: return '2:00 PM'
                        if hh in ['15','3']: return '3:00 PM'
                        if hh in ['16','4']: return '4:00 PM'
                return value
            except Exception:
                return value

        if function_name == "get_available_slots":
            result = self.booking_executor.safe_execute(
                "get_available_slots",
                date=normalize_date(function_args.get("date"))
            )
            try:
                d = normalize_date(function_args.get("date"))
                if d:
                    self.last_slots_date = d
            except Exception:
                pass
            return result
        elif function_name == "book_appointment_slot":
            # Infer date from last slots retrieval if not provided
            inferred_date = function_args.get("date") or self.last_slots_date
            return self.booking_executor.safe_execute(
                "book_appointment_slot",
                date=normalize_date(inferred_date),
                time=normalize_time(function_args.get("time")),
                user_id=function_args.get("user_id")
            )
        else:
            return {
                "status": "error",
                "message": f"Unknown function: {function_name}"
            }

    def generate_response_with_tools(self, user_message: str, system_prompt: str, 
                                   chat_history: List[Dict], user_id: str = None) -> Dict[str, Any]:
        """Generate response with tool calling capabilities"""
        # Direct booking fallback when message includes all needed details
        import re as _re
        def _extract_date(text: str):
            m = _re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
            if m:
                return m.group(0)
            low = text.lower()
            if "today" in low:
                from datetime import datetime
                return datetime.now().date().strftime("%Y-%m-%d")
            if "tomorrow" in low:
                from datetime import datetime, timedelta
                return (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
            return None
        def _extract_time(text: str):
            # try common forms
            m = _re.search(r"\b(0?[1-9]|1[0-6]):00\s*(AM|PM|am|pm)\b", text)
            if m:
                return f"{int(m.group(1))}:00 {m.group(2).upper()}"
            m = _re.search(r"\b(09|10|11|12|13|14|15|16):00\b", text)
            if m:
                mapping = {'09':'9:00 AM','10':'10:00 AM','11':'11:00 AM','12':'12:00 PM','13':'1:00 PM','14':'2:00 PM','15':'3:00 PM','16':'4:00 PM'}
                return mapping.get(m.group(1))
            m = _re.search(r"\b(9|10|11|12|1|2|3|4)\s*(?:o'clock)?\s*(AM|PM|am|pm)\b", text)
            if m:
                return f"{int(m.group(1))}:00 {m.group(2).upper()}"
            m = _re.search(r"\b(9|10|11|12|1|2|3|4)\s*(pm|am)\b", text, _re.IGNORECASE)
            if m:
                return f"{int(m.group(1))}:00 {m.group(2).upper()}"
            return None
        def _extract_user_id(text: str):
            m = _re.search(r"user\s*id\s*[:\-]?\s*([A-Za-z0-9\-]+)", text, _re.IGNORECASE)
            if m:
                return m.group(1)
            m = _re.search(r"\b[A-Za-z0-9]+(?:\-[A-Za-z0-9]+)+\b", text)
            if m:
                return m.group(0)
            return None

        date_candidate = _extract_date(user_message)
        time_candidate = _extract_time(user_message)
        user_candidate = _extract_user_id(user_message)
        if date_candidate and time_candidate and (user_candidate or user_id):
            direct = self.booking_executor.safe_execute(
                "book_appointment_slot",
                date=date_candidate,
                time=time_candidate,
                user_id=(user_candidate or user_id or 'guest')
            )
            status = direct.get("status")
            if status == 'success':
                return {"response": direct.get("message") or "Appointment confirmed.", "tool_used": "book_appointment_slot", "tool_result": direct, "mode": "tool_calling"}
            else:
                # Fall back to tool flow for richer guidance
                pass
        
        # Prepare conversation history
        gemini_history = []
        for msg in chat_history:
            role = 'user' if msg.get("role") == 'user' else 'model'
            text = msg.get("text")
            if text:
                gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

        # Add current user message
        gemini_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

        # Add user context to system prompt
        enhanced_system_prompt = f"""
{system_prompt}

You have access to appointment booking functions. When users want to book appointments:
1. First get available slots for the requested date
2. Present the available options to the user
3. When they confirm a slot, book it for them
4. Always be conversational and helpful

Current user ID: {user_id or 'guest'}
"""

        try:
            # Generate response with tools
            response = self.client.models.generate_content(
                model=CHAT_MODEL,
                contents=gemini_history,
                config=types.GenerateContentConfig(
                    system_instruction=enhanced_system_prompt,
                    tools=self.tools,
                    temperature=0.7
                )
            )

            # Check for function calls
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call'):
                        # Execute the function call
                        function_result = self.execute_tool_call(part.function_call)
                        
                        # Add function call and result to history
                        # Properly re-inject the function_call into history
                        try:
                            gemini_history.append(types.Content(
                                role="model",
                                parts=[types.Part(function_call=part.function_call)]
                            ))
                        except Exception:
                            # Fallback: append the original part if constructor fails
                            gemini_history.append(types.Content(
                                role="model",
                                parts=[part]
                            ))
                        gemini_history.append(types.Content(
                            role="user", 
                            parts=[types.Part.from_function_response(
                                name=part.function_call.name,
                                response=function_result
                            )]
                        ))
                        
                        # Generate follow-up response with function result
                        follow_up_response = self.client.models.generate_content(
                            model=CHAT_MODEL,
                            contents=gemini_history,
                            config=types.GenerateContentConfig(
                                system_instruction=enhanced_system_prompt,
                                temperature=0.7
                            )
                        )
                        # Prefer structured tool result for clarity
                        status = function_result.get("status")
                        if status == 'success':
                            msg = function_result.get("message") or "Appointment updated."
                            try:
                                import re as _re
                                uid = function_call.args.get('user_id')
                                if isinstance(uid, str) and uid:
                                    # collapse duplicated user_id like 'UIDUID' into single 'UID'
                                    msg = _re.sub(fr'({_re.escape(uid)})\1', r'\1', msg)
                            except Exception:
                                pass
                            model_text = msg
                        else:
                            model_text = function_result.get("message") or "There was a problem with your request."

                        return {
                            "response": model_text,
                            "tool_used": part.function_call.name,
                            "tool_result": function_result,
                            "mode": "tool_calling"
                        }

            # Regular response without tools
            # Safely extract text from the initial response
            model_text = getattr(response, "text", None)
            if not model_text:
                try:
                    if getattr(response, "candidates", None):
                        candidate = response.candidates[0]
                        parts = getattr(candidate, "content", None)
                        if parts and getattr(parts, "parts", None):
                            maybe_part = parts.parts[0]
                            model_text = getattr(maybe_part, "text", None)
                except Exception:
                    model_text = None

            if not model_text:
                model_text = "I'm sorry, I couldn't generate a response."

            return {
                "response": model_text,
                "tool_used": None,
                "tool_result": None,
                "mode": "regular"
            }

        except Exception as e:
            print(f"Error in generate_response_with_tools: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "tool_used": None,
                "tool_result": None,
                "mode": "error",
                "error": str(e)
            }

    @staticmethod
    def get_rag_prompt(user_message, context):
        """Standardized RAG prompt template."""
        return f"""
You are a helpful and concise assistant. Use the following CONTEXT to answer the user's question.
If the CONTEXT does not contain the answer, say you cannot answer based on the provided documents.

CONTEXT:
{context}

USER QUESTION: {user_message}
"""

# ======================================================
# 🛠️ Ingestion Handler
# ======================================================

class IngestionHandler:
    """Handles the transformation and validation of text/link data into Document objects."""

    @staticmethod
    def process_text_entries(entries):
        """
        Validates and converts text entries into a list of Document objects.
        Ensures no duplicate names or values.
        """
        seen_names = set()
        seen_values = set()
        documents = []

        duplicate_names = []
        duplicate_values = []

        for entry in entries:
            name = entry.get("name")
            value = entry.get("value")

            if not name or not value:
                raise ValueError("Each text entry must have a name and value.")

            if name in seen_names:
                duplicate_names.append(name)
            else:
                seen_names.add(name)

            if value in seen_values:
                duplicate_values.append(value)
            else:
                seen_values.add(value)

            documents.append(Document(page_content=value, metadata={"name": name, "source_type": "text"}))

        # Raise errors if duplicates exist
        errors = []
        if duplicate_names:
            errors.append(f"Duplicate names detected: {', '.join(duplicate_names)}")
        if duplicate_values:
            errors.append(f"Duplicate values detected: {', '.join(duplicate_values)}")
        if errors:
            raise ValueError(" | ".join(errors))

        return documents

    @staticmethod
    def process_link_entries(entries, rag_bot: 'RagBot'):
        """
        Validates link entries and fetches content via RagBot.
        Ensures no duplicate names or links.
        """
        seen_names = set()
        seen_links = set()
        documents = []

        duplicate_names = []
        duplicate_links = []

        for entry in entries:
            name = entry.get("name")
            link = entry.get("link")

            if not name or not link:
                raise ValueError("Each link entry must have a name and link.")

            if name in seen_names:
                duplicate_names.append(name)
            else:
                seen_names.add(name)

            if link in seen_links:
                duplicate_links.append(link)
            else:
                seen_links.add(link)

            doc = rag_bot.fetch_link(link, name)
            if doc:
                documents.append(doc)

        # Raise errors if duplicates exist
        errors = []
        if duplicate_names:
            errors.append(f"Duplicate names detected: {', '.join(duplicate_names)}")
        if duplicate_links:
            errors.append(f"Duplicate links detected: {', '.join(duplicate_links)}")
        if errors:
            raise ValueError(" | ".join(errors))

        return documents
