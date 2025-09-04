# Hướng dẫn Triển khai Chatbot Trợ Giảng AI

Chào mừng bạn đến với dự án **Chatbot Trợ Giảng AI** – một trợ lý ảo giúp hỗ trợ giáo viên và học sinh trong việc hỏi đáp dựa trên tài liệu học tập. Dưới đây là hướng dẫn chi tiết, từng bước để một người mới cũng có thể thiết lập và triển khai ứng dụng chatbot này.

## Cấu trúc Dự án

Dự án gồm các tệp tin chính sau đây:

- **app.py**: Tệp chính của ứng dụng Streamlit (giao diện web) – kết hợp cả chế độ giáo viên và học sinh. (Tệp này bạn sẽ tạo theo hướng dẫn, sử dụng các hàm từ `common.py` và `kb.py`).
- **common.py**: Chứa các hàm dùng chung, bao gồm cấu hình API của Google Gemini, hàm tạo _embedding_ (vector hóa nội dung) và hàm sinh câu trả lời từ mô hình Gemini.
- **kb.py**: (_Knowledge Base_) Chứa các hàm để quản lý kho kiến thức:
  - Đọc tài liệu đầu vào (.pdf, .docx, .txt).
  - Chia nhỏ văn bản thành các đoạn (chunk).
  - Tạo và lưu trữ vector biểu diễn nội dung của các đoạn văn vào tệp (theo từng lớp/chủ đề).
  - Tìm kiếm những đoạn kiến thức liên quan nhất để hỗ trợ trả lời câu hỏi.
- **requirements.txt**: Danh sách các thư viện Python cần cài đặt để chạy ứng dụng.
- **README.md**: Tài liệu hướng dẫn này.

## Tạo Tài khoản Google Gemini và Lấy API Key

Để sử dụng mô hình ngôn ngữ **Google Gemini**, bạn cần có khóa API (API key) của dịch vụ Google Generative AI. Thực hiện theo các bước sau:

1. Truy cập **Google AI Studio** tại địa chỉ: `https://aistudio.google.com/` (hoặc `https://ai.google.dev`). Đăng nhập bằng tài khoản Google của bạn. Nếu chưa có tài khoản, hãy đăng ký một tài khoản Google miễn phí.
2. Sau khi đăng nhập, ở menu bên trái, tìm và nhấp vào mục **"Get API Key"** (Lấy API key). 
3. Nhấn nút **"Create API Key"** (Tạo API key mới). 
   - Nếu được yêu cầu lựa chọn dự án, chọn **"Generative Language (Gemini) API"** hoặc **"Generative Language Client"** tùy theo giao diện.
   - Xác nhận tạo khóa API. Hệ thống sẽ sinh ra một chuỗi khóa API. **Lưu lại chuỗi này** ở nơi an toàn (khóa API chỉ hiển thị một lần duy nhất).
4. Bạn đã có một API Key cho dịch vụ Gemini. Chuỗi này sẽ được sử dụng trong ứng dụng của chúng ta để gọi mô hình AI.

## Chuẩn bị Môi trường và Mã Nguồn

Trước khi triển khai lên Streamlit Cloud, bạn nên chuẩn bị mã nguồn trên máy hoặc trong một repository GitHub:

1. **Tạo file dự án**: Tạo một thư mục dự án trên máy của bạn. Bên trong, tạo các tệp tin: `app.py`, `common.py`, `kb.py`, `requirements.txt`, `README.md` và sao chép nội dung tương ứng (đã cung cấp ở phần trên) vào các file này.
2. **Cài đặt thư viện (chạy local)**: Nếu muốn chạy thử trên máy, hãy chắc chắn bạn đã cài Python 3.9+ và sử dụng `pip` để cài các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
