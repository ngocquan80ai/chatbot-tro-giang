"""
common.py
- Không ném Exception khi import nếu thiếu API key (để app còn render UI).
- Lấy GEMINI_API_KEY từ st.secrets (ưu tiên) hoặc biến môi trường.
- Cung cấp 2 hàm:
    embed_texts(texts: list[str]) -> np.ndarray
    generate_answer(question: str, context: list[str] | str | None = None) -> str
"""

from __future__ import annotations
import os
import numpy as np

# Thử import streamlit để đọc secrets khi chạy trên Streamlit Cloud
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None

# SDK Gemini
import google.generativeai as genai

_CONFIGURED = False  # đã configure API hay chưa

def _get_api_key() -> str | None:
    """Ưu tiên lấy từ st.secrets, sau đó biến môi trường."""
    if st is not None:
        try:
            val = st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
            if val:
                return val
        except Exception:
            pass
    return os.getenv("GEMINI_API_KEY")

def _ensure_config() -> bool:
    """Configure SDK đúng 1 lần. Trả về True nếu đã có key, False nếu thiếu."""
    global _CONFIGURED
    if _CONFIGURED:
        return True
    api_key = _get_api_key()
    if not api_key:
        # KHÔNG raise ở đây: để app vẫn lên giao diện.
        if st is not None:
            st.warning("⚠️ Chưa thiết lập GEMINI_API_KEY (Settings → Secrets). Một số chức năng sẽ không hoạt động.")
        else:
            print("[warn] GEMINI_API_KEY missing")
        return False
    genai.configure(api_key=api_key)
    _CONFIGURED = True
    return True

# ========= EMBEDDINGS =========
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Tạo embedding cho danh sách chuỗi. Trả về mảng (N, D).
    Nếu thiếu API key → raise RuntimeError (tại thời điểm dùng).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if not _ensure_config():
        raise RuntimeError("GEMINI_API_KEY chưa được thiết lập trong Secrets/ENV.")

    vecs = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        # model embedding mới
        resp = genai.embed_content(
            model="models/text-embedding-004",
            content=t,
            task_type="retrieval_document",
        )
        # Chuẩn hóa lấy vector
        emb = None
        if isinstance(resp, dict):
            emb = resp.get("embedding")
            if isinstance(emb, dict):
                emb = emb.get("values") or emb.get("value")
            if emb is None and "embeddings" in resp:
                try:
                    emb = resp["embeddings"][0].get("values") or resp["embeddings"][0].get("value")
                except Exception:
                    emb = None
        else:
            try:
                emb_obj = getattr(resp, "embedding", None)
                emb = getattr(emb_obj, "values", None) or getattr(emb_obj, "value", None)
            except Exception:
                emb = None

        if emb is None:
            continue
        vecs.append(np.array(emb, dtype=np.float32))

    if not vecs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(vecs)

# ========= GENERATION =========
def generate_answer(
    question: str,
    context: list[str] | str | None = None
) -> str:
    """
    Gọi Gemini-1.5-Flash sinh trả lời. Nếu có context (1 hoặc nhiều đoạn), mô hình sẽ
    được hướng dẫn chỉ bám vào context. Nếu thiếu API key → thông báo gọn.
    """
    if not _ensure_config():
        return "⚠️ Ứng dụng chưa có GEMINI_API_KEY (Settings → Secrets)."

    # Chuẩn bị prompt
    if context:
        ctx = "\n".join(context) if isinstance(context, list) else str(context)
        system = (
            "Bạn là Trợ giảng AI. Chỉ sử dụng thông tin trong 'Tài liệu tham khảo' để trả lời. "
            "Nếu không đủ thông tin, hãy nói: 'Không có trong tài liệu'."
        )
        prompt = (
            f"{system}\n\n"
            f"Tài liệu tham khảo (trích đoạn):\n{ctx}\n\n"
            f"Câu hỏi: {question}\n"
            f"Hãy trả lời ngắn gọn, chính xác, bám sát tài liệu."
        )
    else:
        prompt = f"Câu hỏi: {question}\nTrả lời ngắn gọn, chính xác bằng tiếng Việt."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if text and text.strip():
            return text.strip()
        # fallback khi SDK thay đổi cấu trúc
        try:
            parts = []
            for c in resp.candidates:  # type: ignore[attr-defined]
                for p in c.content.parts:
                    if hasattr(p, "text"):
                        parts.append(p.text)
            if parts:
                return "\n".join(parts).strip()
        except Exception:
            pass
        return "Không có trong tài liệu."
    except Exception as e:
        if st is not None:
            st.error(f"Lỗi gọi mô hình: {e}")
        else:
            print(f"[Generation error] {e}")
        return "Đã xảy ra lỗi khi gọi mô hình."
