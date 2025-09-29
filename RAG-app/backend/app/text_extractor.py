from __future__ import annotations

from io import BytesIO
from typing import List


def extract_text_from_pdf(content: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return content.decode("utf-8", errors="ignore")
    with BytesIO(content) as bio:
        return extract_text(bio) or ""


def extract_text_from_docx(content: bytes) -> str:
    try:
        import docx  # type: ignore
    except Exception:
        return content.decode("utf-8", errors="ignore")
    file_like = BytesIO(content)
    doc = docx.Document(file_like)
    parts: List[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)

