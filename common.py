import os

# Đọc khóa API từ biến môi trường hoặc từ streamlit secrets (nếu chạy trên Streamlit)
API_KEY = os.getenv("GEMINI_API_KEY")
try:
    import streamlit as st
    # Nếu có sử dụng Streamlit, lấy API key từ file Secrets
    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
        # Đặt API key vào biến môi trường để SDK của Google có thể tự động sử dụng
        os.environ["GEMINI_API_KEY"] = API_KEY
except Exception as e:
    # Không dùng streamlit hoặc không có secrets, sẽ dùng API_KEY từ env (nếu có)
    pass

# Kiểm tra API key, nếu chưa cấu hình thì báo lỗi để người dùng biết
if not API_KEY:
    raise Exception("Chưa cấu hình GEMINI_API_KEY. Vui lòng thêm API key vào Streamlit Secrets hoặc biến môi trường!")

# Import SDK của Google Gemini
from google import genai

# Khởi tạo client cho API Gemini (sử dụng API key từ môi trường)
client = genai.Client()

# Định nghĩa tên các mô hình sử dụng
EMBED_MODEL = "gemini-embedding-001"   # Mô hình dùng để tạo embedding (vector) cho văn bản
GEN_MODEL = "gemini-2.5-flash"         # Mô hình dùng để sinh nội dung (trả lời câu hỏi)

def create_embedding(content):
    """
    Tạo vector embedding cho văn bản hoặc danh sách văn bản.
    - content: Chuỗi văn bản đầu vào hoặc danh sách các chuỗi.
    - Trả về: Vector (danh sách số thực) cho chuỗi đầu vào, 
              hoặc danh sách các vector nếu đầu vào là danh sách chuỗi.
    """
    try:
        # Nếu content là chuỗi đơn, cho vào list để xử lý đồng bộ
        if isinstance(content, str):
            contents = [content]
        else:
            contents = list(content)  # chuyển sang list để chắc chắn có thể lặp
        
        # Gọi API Gemini để tạo embedding cho các nội dung trong danh sách
        result = client.models.embed_content(model=EMBED_MODEL, contents=contents)
        # Kết quả trả về chứa thuộc tính embeddings (danh sách các vector)
        # Mỗi phần tử trong result.embeddings có thuộc tính values là list số float
        vectors = [emb.values for emb in result.embeddings]
        
        # Nếu đầu vào chỉ một chuỗi, trả về vector (thay vì list chứa một phần tử)
        if isinstance(content, str):
            return vectors[0] if vectors else []
        # Nếu đầu vào là list chuỗi, trả về danh sách vector tương ứng
        return vectors
    except Exception as e:
        # In ra lỗi (nếu có) để tiện gỡ lỗi
        print("Lỗi khi tạo embedding:", e)
        try:
            import streamlit as st
            st.error("Lỗi: Không thể tạo embedding cho văn bản.")
        except:
            pass
        return []  # Trả về list rỗng nếu có lỗi

def generate_answer(question, context=None):
    """
    Sinh câu trả lời cho câu hỏi dựa trên kiến thức (context) cho trước bằng mô hình Gemini.
    - question: Câu hỏi đầu vào (chuỗi văn bản).
    - context: Nội dung tri thức tham khảo (chuỗi hoặc danh sách các đoạn văn bản). Có thể là None nếu không có.
    - Trả về: Câu trả lời (chuỗi văn bản).
    """
    try:
        # Chuẩn bị prompt cho mô hình
        if context:
            # Nếu có context, ghép context vào prompt để mô hình sử dụng làm tư liệu
            # Nếu context là danh sách các đoạn, kết hợp thành một khối văn bản
            context_text = "\n".join(context) if isinstance(context, list) else str(context)
            prompt = (f"Tài liệu tham khảo:\n{context_text}\n\n"
                      f"Câu hỏi: {question}\n"
                      "Hãy dựa vào tài liệu trên để trả lời bằng tiếng Việt thật chi tiết và dễ hiểu.")
        else:
            # Nếu không có context, chỉ đưa câu hỏi và yêu cầu trả lời
            prompt = (f"Câu hỏi: {question}\n"
                      "Trả lời bằng kiến thức của bạn bằng tiếng Việt một cách chi tiết và dễ hiểu.")
        
        # Gọi API Gemini để sinh nội dung trả lời
        response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
        answer = response.text  # kết quả trả lời từ mô hình (chuỗi văn bản)
        return answer
    except Exception as e:
        # Xử lý ngoại lệ nếu gọi API thất bại
        print("Lỗi khi gọi API Gemini:", e)
        try:
            import streamlit as st
            st.error("Đã xảy ra lỗi khi sinh câu trả lời từ mô hình Gemini.")
        except:
            pass
        # Trả về câu xin lỗi người dùng
        return "Xin lỗi, tôi không thể trả lời câu hỏi này vào lúc này."
