# kb.py — bản chuẩn cho app gộp (không phụ thuộc common.py)
from __future__ import annotations
import os
import io
import json
import numpy as np
from typing import List, Tuple

from pypdf import PdfReader          # đọc PDF
from docx import Document            # đọc DOCX

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def slugify_name(name: str) -> str:
    import re
    name = (name or "").strip()
    # giữ chữ cái (kể cả có dấu), số, _ và -
    out = []
    for ch in name:
        if ch.isalnum() or ch in "_-":
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    s = "".join(out)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "untitled"

def read_document(uploaded_file) -> str:
    """
    Đọc nội dung từ UploadedFile (.pdf/.docx/.txt) và trả về chuỗi.
    Trả về chuỗi 'Lỗi …' nếu có lỗi để UI hiển thị rõ.
    """
    try:
        fname = getattr(uploaded_file, "name", "").lower()
        ext = os.path.splitext(fname)[1]
        if ext == ".pdf":
            data = uploaded_file.read()
            reader = PdfReader(io.BytesIO(data))
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text).strip()
        elif ext == ".docx":
            # python-docx nhận file-like; nếu không được thì chuyển sang BytesIO
            try:
                doc = Document(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                doc = Document(io.BytesIO(uploaded_file.read()))
            return "\n".join(p.text for p in doc.paragraphs).strip()
        elif ext == ".txt":
            b = uploaded_file.read()
            return (b.decode("utf-8", errors="ignore")).strip()
        else:
            return "Lỗi: Định dạng không hỗ trợ. Hãy dùng PDF/DOCX/TXT."
    except Exception as e:
        return f"Lỗi khi đọc tài liệu: {e}"

def split_into_chunks(text: str, max_words: int = 100) -> List[str]:
    """
    Chia văn bản thành các đoạn ~max_words từ (để tìm kiếm và prompt hiệu quả).
    """
    import re
    if not text or not text.strip():
        return []
    text = text.replace("\r", " ").replace("\n", " ")
    sentences = re.split(r'(?<=[\.!?])\s+', text)
    chunks, cur, cur_len = [], [], 0
    for sent in sentences:
        words = sent.split()
        while len(words) > max_words:          # câu quá dài thì cắt nhỏ
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if not words:
            continue
        if cur_len + len(words) <= max_words:
            cur.extend(words)
            cur_len += len(words)
        else:
            chunks.append(" ".join(cur))
            cur, cur_len = words, len(words)
    if cur:
        chunks.append(" ".join(cur))
    return [c.strip() for c in chunks if c.strip()]

def _base_paths(class_code: str, topic_slug: str) -> Tuple[str, str]:
    cls = slugify_name(class_code)
    t   = slugify_name(topic_slug)
    base = f"{cls}_{t}"
    return os.path.join(DATA_DIR, base + ".json"), os.path.join(DATA_DIR, base + ".npy")

def save_knowledge(class_code: str, topic_slug: str, chunks: List[str], embeddings: np.ndarray):
    """
    Lưu chunks (JSON) và embeddings (NPY) theo mã lớp + slug chủ đề.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    text_fp, emb_fp = _base_paths(class_code, topic_slug)
    try:
        with open(text_fp, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        np.save(emb_fp, embeddings)
        return True
    except Exception as e:
        return f"Lỗi lưu tri thức: {e}"

def load_knowledge(class_code: str, topic_slug: str) -> Tuple[List[str] | None, np.ndarray | None]:
    """
    Tải lại (chunks, embeddings) cho lớp + chủ đề. Không có thì trả (None, None).
    """
    text_fp, emb_fp = _base_paths(class_code, topic_slug)
    if not (os.path.exists(text_fp) and os.path.exists(emb_fp)):
        return None, None
    try:
        with open(text_fp, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        emb = np.load(emb_fp)
        return chunks, emb
    except Exception:
        return None, None
