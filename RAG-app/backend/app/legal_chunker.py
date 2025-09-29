from __future__ import annotations

import re
from typing import Dict, List, Any


CHUONG_RE = re.compile(r"^\s*Ch\u01b0\u01a1ng\s+([IVXLCDM]+|\d+)\s*[:\-.]?\s*(.*)$", re.IGNORECASE)
MUC_RE = re.compile(r"^\s*M\u1ee5c\s+(\d+)\s*[:\-.]?\s*(.*)$", re.IGNORECASE)
DIEU_RE = re.compile(r"^\s*\u0110i\u1ec1u\s+(\d+)\.?\s*(.*)$", re.IGNORECASE)
KHOAN_RE = re.compile(r"^\s*Kho\u1ea3n\s+(\d+)\s*[:\-\.)]?\s*(.*)$", re.IGNORECASE)
DIEM_RE = re.compile(r"^\s*\u0110i\u1ec3m\s+([a-zA-Z])\s*[:\-\.)]?\s*(.*)$", re.IGNORECASE)


def _normalize_text(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n")]
    return [ln for ln in lines if ln]


def chunk_legal_text(text: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines = _normalize_text(text)
    chunks: List[Dict[str, Any]] = []

    current_chuong_num: str | None = None
    current_chuong_title: str | None = None
    current_muc_num: str | None = None
    current_muc_title: str | None = None
    current_dieu_num: str | None = None
    current_dieu_title: str | None = None
    buffer: List[str] = []

    def flush_dieu() -> None:
        nonlocal buffer, current_dieu_num, current_dieu_title
        if current_dieu_num is None:
            buffer = []
            return
        dieu_text = "\n".join(buffer).strip()
        if not dieu_text:
            buffer = []
            return
        metadata = {
            **base_metadata,
            "chuong_number": current_chuong_num,
            "chuong_title": current_chuong_title,
            "muc_number": current_muc_num,
            "muc_title": current_muc_title,
            "dieu_number": int(current_dieu_num) if current_dieu_num and current_dieu_num.isdigit() else current_dieu_num,
            "heading": current_dieu_title or "",
        }
        # Chia theo Khoản và luôn tạo 1 chunk/ Khoản (bao gồm toàn bộ nội dung Khoản và các Điểm bên trong).
        khoan_splits = split_by_khoan(dieu_text)
        for khoan_number, khoan_text in khoan_splits:
            meta2 = {**metadata, "khoan_number": khoan_number}
            header_lines: List[str] = []
            if meta2.get("dieu_number") is not None or meta2.get("heading"):
                header_lines.append(f"Điều {meta2.get('dieu_number')}. {meta2.get('heading')}")
            if khoan_number is not None:
                header_lines.append(f"Khoản {khoan_number}")
            composed = ("\n".join(header_lines) + ("\n" if header_lines else "") + khoan_text).strip()
            chunks.append({"text": composed, "metadata": meta2})
        buffer = []

    for ln in lines:
        if m := CHUONG_RE.match(ln):
            flush_dieu()
            current_chuong_num = m.group(1)
            current_chuong_title = m.group(2).strip()
            current_muc_num = None
            current_muc_title = None
            continue
        if m := MUC_RE.match(ln):
            flush_dieu()
            current_muc_num = m.group(1)
            current_muc_title = m.group(2).strip()
            continue
        if m := DIEU_RE.match(ln):
            flush_dieu()
            current_dieu_num = m.group(1)
            current_dieu_title = m.group(2).strip()
            continue
        buffer.append(ln)

    flush_dieu()
    return chunks


def split_by_khoan(text: str) -> List[tuple[int, str]]:
    lines = text.split("\n")
    segments: List[tuple[int, List[str]]] = []
    current_num: int | None = None
    current_buf: List[str] = []
    for ln in lines:
        if m := KHOAN_RE.match(ln):
            if current_num is not None:
                segments.append((current_num, current_buf))
            try:
                current_num = int(m.group(1))
            except Exception:
                current_num = None
            current_buf = [m.group(0)]
        else:
            current_buf.append(ln)
    if current_buf:
        segments.append((current_num or 0, current_buf))
    if len(segments) <= 1 and (current_num is None):
        return [(0, text)]
    return [(num, "\n".join(buf).strip()) for num, buf in segments]


def split_by_diem(text: str) -> List[tuple[str, str]]:
    lines = text.split("\n")
    segments: List[tuple[str, List[str]]] = []
    current_letter: str | None = None
    current_buf: List[str] = []
    for ln in lines:
        if m := DIEM_RE.match(ln):
            if current_letter is not None:
                segments.append((current_letter, current_buf))
            current_letter = m.group(1)
            current_buf = [m.group(0)]
        else:
            current_buf.append(ln)
    if current_buf:
        segments.append((current_letter or "", current_buf))
    if len(segments) <= 1 and (current_letter is None):
        return [("", text)]
    return [(letter, "\n".join(buf).strip()) for letter, buf in segments]

