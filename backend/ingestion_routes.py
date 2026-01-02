import json
import logging
from flask import Blueprint, request, jsonify, current_app
from rag_core import IngestionHandler
from utils import get_file_loader

# -----------------------------------------------------
# 📦 Define Blueprint
# -----------------------------------------------------
ingestion_bp = Blueprint('ingestion', __name__)

# Set up logger
logger = logging.getLogger(__name__)

# ======================================================
# 🔗 INGEST ALL DATA (Unified Endpoint)
# ======================================================
@ingestion_bp.route("/all", methods=["POST"])
def ingest_all():
    """
    Unified ingestion endpoint: processes text entries, link entries, and file uploads.
    Handles any combination of types and only resets vector store if there is new data.
    """
    rag_system = current_app.config['RAG_SYSTEM']

    # --- 1️⃣ Parse incoming data ---
    data = request.form.get("data")
    if data:
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON in 'data' field."}), 400
    else:
        try:
            data = request.get_json() or {}
        except Exception:
            data = {}

    all_documents = []
    total_count = 0
    sources_processed = []

    # --- 2️⃣ Process Text Entries ---
    text_entries = data.get("textEntries", [])
    if text_entries:
        try:
            text_docs = IngestionHandler.process_text_entries(text_entries)
            if text_docs:
                all_documents.extend(text_docs)
                total_count += len(text_docs)
                sources_processed.append("Text")
        except Exception as e:
            logger.warning(f"Error processing text entries: {e}")

    # --- 3️⃣ Process Link Entries ---
    link_entries = data.get("linkEntries", [])
    if link_entries:
        try:
            link_docs = IngestionHandler.process_link_entries(link_entries, rag_system)
            if link_docs:
                all_documents.extend(link_docs)
                total_count += len(link_docs)
                sources_processed.append("Links")
        except Exception as e:
            logger.warning(f"Error processing link entries: {e}")

    # --- 4️⃣ Process File Uploads ---
    uploaded_files = request.files.getlist("files")
    if uploaded_files:
        try:
            file_docs = []
            processed_file_count = 0
            for file in uploaded_files:
                docs, count = get_file_loader(file)
                if docs:
                    file_docs.extend(docs)
                    processed_file_count += count
            if file_docs:
                all_documents.extend(file_docs)
                total_count += processed_file_count
                sources_processed.append("Files")
        except Exception as e:
            logger.warning(f"Error processing uploaded files: {e}")

    # --- 5️⃣ Check if any documents were processed ---
    if not all_documents:
        return jsonify({"message": "No new data found. RAG knowledge base remains unchanged."}), 200

    # --- 6️⃣ Reset vector store before storing new data ---
    rag_system.vector_store = None

    # --- 7️⃣ Store all documents ---
    source_summary = ", ".join(sources_processed) or "Various"
    try:
        return rag_system.store_documents(
            all_documents,
            source=source_summary,
            count=total_count,
        )
    except Exception as e:
        logger.error(f"Failed to store documents: {e}")
        return jsonify({"error": f"Failed to store documents: {str(e)}"}), 500

# ======================================================
# 🔄 RESET KNOWLEDGE BASE
# ======================================================
@ingestion_bp.route("/reset", methods=["POST"])
def reset_knowledge_base():
    """
    Clears the current vector store, effectively removing all ingested data.
    """
    try:
        rag_system = current_app.config['RAG_SYSTEM']
        rag_system.vector_store = None
        logger.info("Knowledge base reset successful.")
        return jsonify({"message": "RAG knowledge base successfully reset and cleared."}), 200
    except Exception as e:
        logger.error(f"Failed to reset knowledge base: {e}")
        return jsonify({"error": f"Failed to reset knowledge base: {str(e)}"}), 500
