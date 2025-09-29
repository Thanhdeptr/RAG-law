from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    text: str = Field(..., description="Raw legal text to ingest")
    doc_metadata: Optional[Dict[str, Any]] = Field(default=None)


class IngestResult(BaseModel):
    chunks_added: int


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]

