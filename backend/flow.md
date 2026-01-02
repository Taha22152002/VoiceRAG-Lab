               ┌───────────────────────┐
               │      Client / UI      │
               │  (Browser / Frontend) │
               └─────────┬─────────────┘
                         │ HTTP POST
                         │ /ingest/all or /chat
                         ▼
               ┌───────────────────────┐
               │      Flask App        │
               │   (app.py factory)    │
               ├─────────┬─────────────┤
               │ Config: RAG_SYSTEM    │
               │ Config: EMBEDDING_MODEL│
               │ Config: CHAT_MODEL    │
               └─────────┬─────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
 ┌─────────────────┐            ┌──────────────────────┐
 │Ingestion Routes  │            │     Chat Endpoint    │
 │(ingestion_bp)    │            │      /chat           │
 │ /all, /reset     │            └──────────────────────┘
 └─────────┬────────┘
           │
           ▼
 ┌─────────────────────────────┐
 │   IngestionHandler / Utils  │
 │                             │
 │ - process_text_entries      │
 │ - process_link_entries      │
 │ - get_file_loader           │
 │ - parse_grounding_metadata  │
 └─────────┬───────────────────┘
           │
           ▼
 ┌─────────────────────────────┐
 │          RagBot             │
 │  (Core RAG / LLM Logic)    │
 │                             │
 │ - store_documents()         │
 │ - get_rag_context()         │
 │ - fetch_link()              │
 │ - generate_response()       │
 │                             │
 │  Internal: Vector Store     │
 │    (Chroma, embeddings)     │
 └─────────┬───────────────────┘
           │
           ▼
 ┌─────────────────────────────┐
 │       Vector Store          │
 │   (Chroma Collection)       │
 │ - stores embedded document  │
 │ - supports retrieval        │
 └─────────────────────────────┘


---------------------------------------------------
API ENDPOINT

User → Gemini (interprets intent) → JSON command
      ↓
Tool router (Python executes GET/POST)
      ↓
Result → Gemini (for natural confirmation wording)
