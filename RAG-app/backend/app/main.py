from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .embeddings import OllamaEmbeddingsClient, OllamaChatClient
from .vectorstore import ChromaStore
from .legal_chunker import chunk_legal_text
from .schemas import IngestTextRequest, IngestResult, QueryRequest, QueryResponse
from .text_extractor import extract_text_from_pdf, extract_text_from_docx


app = FastAPI(title="Legal RAG for Vietnamese Docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


embedder = OllamaEmbeddingsClient(settings.ollama_base, settings.embedding_model)
chat = OllamaChatClient(settings.ollama_base, settings.generation_model)
store = ChromaStore(settings.chroma_persist_dir)


@app.post("/ingest/text", response_model=IngestResult)
def ingest_text(req: IngestTextRequest) -> IngestResult:
    chunks = chunk_legal_text(req.text, req.doc_metadata or {})
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    store.add_texts(texts=texts, metadatas=metadatas, embeddings=embeddings)
    return IngestResult(chunks_added=len(texts))


@app.post("/ingest/file", response_model=IngestResult)
def ingest_file(file: UploadFile = File(...), doc_title: Optional[str] = Form(None)) -> IngestResult:
    content = file.file.read()
    text: str
    if file.filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(content)
    elif file.filename.lower().endswith(".docx"):
        text = extract_text_from_docx(content)
    else:
        text = content.decode("utf-8", errors="ignore")

    base_meta = {"file_name": file.filename}
    if doc_title:
        base_meta["doc_title"] = doc_title

    chunks = chunk_legal_text(text, base_meta)
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    store.add_texts(texts=texts, metadatas=metadatas, embeddings=embeddings)
    return IngestResult(chunks_added=len(texts))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    q_embed = embedder.embed_texts([req.question])[0]
    results = store.query_by_embedding(q_embed, top_k=req.top_k or 5)
    contexts: List[str] = []
    citations: List[dict] = []
    for doc, meta in zip(results.documents, results.metadatas):
        heading = meta.get("heading", "")
        dieu = meta.get("dieu_number")
        khoan = meta.get("khoan_number")
        diem = meta.get("diem_letter")
        label_parts: List[str] = []
        if dieu is not None:
            label_parts.append(f"Điều {dieu}")
        if khoan is not None:
            label_parts.append(f"Khoản {khoan}")
        if diem is not None:
            label_parts.append(f"Điểm {diem}")
        label = ", ".join(label_parts)
        prefix = f"[{label}] {heading}" if label or heading else ""
        contexts.append(f"{prefix}\n{doc}")
        citations.append({
            "label": label,
            "heading": heading,
            "metadata": meta,
        })

    system_msg = {
        "role": "system",
        "content": (
            "Bạn là trợ lý pháp lý. Trả lời bằng tiếng Việt, ngắn gọn, chính xác. "
            "Luôn trích dẫn Điều/Khoản/Điểm liên quan. Nếu không chắc, nói rõ không tìm thấy."
        ),
    }
    context_block = "\n\n--- NGỮ CẢNH ---\n" + "\n\n".join(contexts)
    user_msg = {"role": "user", "content": f"Câu hỏi: {req.question}{context_block}"}
    answer = chat.chat([system_msg, user_msg], temperature=req.temperature or 0.1, max_tokens=req.max_tokens)
    return QueryResponse(answer=answer, citations=citations)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "ollama_base": settings.ollama_base,
        "generation_model": settings.generation_model,
        "embedding_model": settings.embedding_model,
    }

