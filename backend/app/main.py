"""ClearPath Chatbot - FastAPI Application Entry Point.

Handles:
  - CORS configuration for frontend
  - Static file serving for the frontend
  - Startup event: loads PDFs, builds/loads FAISS index
  - Mounts all API routes
"""
import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.api.routes import router as api_router
from backend.app.config import DOCS_DIR, PORT
from backend.app.rag.pdf_loader import load_all_documents
from backend.app.rag.chunker import chunk_documents
from backend.app.rag.retriever import retriever

# -- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# -- FastAPI app ---
app = FastAPI(
    title="ClearPath Chatbot",
    description="Customer support chatbot with RAG pipeline, model routing, and output evaluation",
    version="1.0.0",
)

# CORS - allow frontend dev server and same-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

# Serve frontend static files
_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html."""
    index_path = _frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "ClearPath Chatbot API", "docs": "/docs"}


# -- Startup: Build or load RAG index ---

@app.on_event("startup")
async def startup_event():
    """Load documents and build/load the FAISS index on server start."""
    logger.info("=" * 60)
    logger.info("ClearPath Chatbot - Starting up")
    logger.info("=" * 60)

    # Try loading cached index first
    if retriever.load_index():
        logger.info("Using cached FAISS index")
        return

    # Build index from PDFs
    logger.info(f"Building index from PDFs in {DOCS_DIR}")

    if not DOCS_DIR.exists():
        logger.error(f"Docs directory not found: {DOCS_DIR}")
        logger.error("Make sure the 'docs/' folder exists at the project root")
        return

    # Step 1: Extract text from PDFs
    documents = load_all_documents(DOCS_DIR)
    if not documents:
        logger.error("No documents loaded - index will be empty")
        return

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)
    if not chunks:
        logger.error("No chunks generated - index will be empty")
        return

    # Step 3: Build FAISS index
    retriever.build_index(chunks)

    logger.info("=" * 60)
    logger.info("ClearPath Chatbot - Ready!")
    logger.info("=" * 60)
