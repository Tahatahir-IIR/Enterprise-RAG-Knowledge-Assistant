from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import RedirectResponse

from . import __version__
from .auth import User, current_user
from .ingestion.loaders import SUPPORTED_EXTENSIONS
from .models import AskRequest, AskResponse, DocumentInfo, DocumentMetadata
from .pipeline import RagPipeline

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide HuggingFace HEAD requests at startup

_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


def set_pipeline(p: RagPipeline) -> None:
    global _pipeline
    _pipeline = p


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()
    yield
    if _pipeline:
        _pipeline.close()


app = FastAPI(
    title="Dossier - Enterprise RAG",
    version=__version__,
    description="Bilingual (FR/AR) document assistant with cited, grounded answers.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    p = get_pipeline()
    return {
        "status": "ok",
        "version": __version__,
        "documents": len(p.documents),
        "chunks": p.store.count(),
        "llm": p.llm.name,
        "embedding_dim": p.embedder.dim,
        "chunk_strategy": p.chunker.name,
    }


@app.post("/documents", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    department: str = Form(...),
    doc_type: str = Form(...),
    language: str | None = Form(None),
    doc_date: date | None = Form(None),
    tags: str = Form(""),
    user: User = Depends(current_user),
):
    if not user.can_access(department):
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"You cannot upload to department '{department}'")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported type {ext}")
    meta = DocumentMetadata(
        department=department,
        doc_type=doc_type,
        language=language or None,
        doc_date=doc_date,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
    )
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        tmp = tmp_dir / (file.filename or f"upload{ext}")
        with tmp.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        return get_pipeline().ingest_file(tmp, meta, source_name=file.filename)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(user: User = Depends(current_user)):
    return get_pipeline().list_documents(user)


@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, user: User = Depends(current_user)):
    p = get_pipeline()
    info = p.documents.get(doc_id)
    if info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown document")
    if not user.can_access(info.department):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not allowed")
    p.delete_document(doc_id)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, user: User = Depends(current_user)):
    if req.filters.department and not user.can_access(req.filters.department):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You cannot query that department")
    return get_pipeline().ask(req.question, user, req.filters, req.top_k, req.use_cache)


@app.get("/cache/stats")
def cache_stats(user: User = Depends(current_user)):
    return get_pipeline().cache.stats()


@app.delete("/cache", status_code=status.HTTP_204_NO_CONTENT)
def clear_cache(user: User = Depends(current_user)):
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    get_pipeline().cache.clear()
