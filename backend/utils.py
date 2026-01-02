# utils.py

import os
import json
import tempfile
import pandas as pd
from flask import Flask, request, make_response
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

def setup_cors(app: Flask):
    """Configures CORS headers for the Flask app."""
    DEV_ORIGINS = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://[::]:8000",
    ]
    
    # Use flask-cors to add common CORS headers automatically.
    from flask_cors import CORS
    CORS(app, origins=DEV_ORIGINS, supports_credentials=True)

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin")
        if origin and origin in DEV_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin
        else:
            response.headers["Access-Control-Allow-Origin"] = "*" # Fallback for local dev
        # Standard headers are usually handled by flask-cors, but kept for robustness
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        response.headers.setdefault("Access-Control-Max-Age", "3600")
        return response

def get_file_loader(file):
    """Handles file saving, loading, and cleanup for one file."""
    filename = file.filename
    temp_file_path = None
    documents = []
    
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            temp_file_path = tmp.name

        # The core file-type parsing logic
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
            documents.extend(loader.load())
        elif filename.lower().endswith((".docx", ".doc")):
            loader = Docx2txtLoader(temp_file_path)
            documents.extend(loader.load())
        elif filename.lower().endswith(".txt"):
            with open(temp_file_path, "r", encoding="utf-8") as f:
                documents.append(Document(page_content=f.read()))
        elif filename.lower().endswith(".json"):
             # JSON documents are read, loaded, and formatted as a single string document
             with open(temp_file_path, "r", encoding="utf-8") as f:
                 data = json.dumps(json.load(f))
                 documents.append(Document(page_content=data))
        elif filename.lower().endswith((".csv", ".xlsx", ".xls")):
            # Uses pandas for robust handling of tabular data
            df = pd.read_excel(temp_file_path) if filename.lower().endswith((".xlsx", ".xls")) else pd.read_csv(temp_file_path)
            csv_text = df.to_csv(index=False)
            documents.append(Document(page_content=csv_text))
        else:
            print(f"Skipping unsupported file: {filename}")

    except Exception as e:
        print(f"File loading error for {filename}: {e}")
        return None, 0
        
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return documents, len(documents)

def parse_grounding_metadata(response):
    """Safely extracts sources/attributions from the Gemini API response."""
    sources = []
    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            gm = getattr(candidates[0], "grounding_metadata", None)
            if gm:
                # Try multiple possible attribute names for attributions/sources/chunks
                attributions = (
                    getattr(gm, "grounding_attributions", None)
                    or getattr(gm, "attributions", None)
                    or getattr(gm, "sources", None)
                    or getattr(gm, "grounding_chunks", None)
                    or getattr(gm, "chunks", None)
                )
                if attributions:
                    for attr in attributions:
                        # Handle both direct web links and nested chunk structures
                        web = getattr(attr, "web", None) if hasattr(attr, "web") else attr # Assume direct source if 'web' not found
                        uri = getattr(web, "uri", None)
                        title = getattr(web, "title", None)
                        if uri:
                            sources.append({"uri": uri, "title": title or "Source"})
    except Exception as meta_err:
        print(f"Grounding metadata parsing error: {meta_err}")

    return sources