import streamlit as st
import os
import json
import numpy as np
import re

# Import các hàm xử lý từ các module common.py và kb.py
from common import embed_texts, generate_answer
from kb import read_document, split_into_chunks, save_knowledge, load_knowledge

# Thiết lập thư mục lưu trữ dữ liệu
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Hàm tiện ích để chuẩn hóa chuỗi thành tên tập tin an toàn (loại bỏ ký tự đặc biệt, khoảng trắng)
def safe_filename(name: str) -> str:
    # Thay thế các chuỗi ký tự không phải chữ, số, hoặc gạch dưới bằng ký tự gạch dưới "_"
    return re.sub(r'\W+', '_', name.strip())

# Cấu hình trang Streamlit
st.set_page_config(page_title="AI Teaching Assistant Chatbot", layout="wide")

# Tiêu đề ứng dụng
st.title("💬 Chatbot Trợ Giảng AI")

# Sidebar lựa chọn chế độ
mode = st.sidebar.radio("Chọn chế độ:", ("Giáo viên", "Học sinh"))

# Chế độ Giáo viên
if mode == "Giáo viên":
    st.header("📚 Chế độ Giáo viên")
    # Yêu cầu nhập mã PIN để xác thực quyền giáo viên
    pin_input = st.sidebar.text_input("Nhập mã PIN của giáo viên:", type="password")
    if pin_input:
        # Kiểm tra mã PIN
        if st.secrets.get("TEACHER_PIN") is None:
            st.sidebar.error("⚠️ Không tìm thấy TEACHER_PIN trong cấu hình ứng dụng.")
        elif pin_input == st.secrets["TEACHER_PIN"]:
            st.sidebar.success("✅ Đăng nhập thành công! Bạn có thể quản lý lớp học.")
            # Đọc danh sách các lớp đã có (dựa trên các file info trong thư mục DATA_DIR)
            class_codes = []
            for fname in os.listdir(DATA_DIR):
                if fname.endswith("_info.json"):
                    code = fname.replace("_info.json", "")
                    class_codes.append(code)
            class_codes.sort()

            # Giao diện tạo lớp mới
            st.subheader("Tạo lớp mới")
            new_class_code_input = st.text_input("Nhập mã lớp mới:")
            create_btn = st.button("Tạo lớp")
            if create_btn:
                if not new_class_code_input:
                    st.warning("Vui lòng nhập mã lớp hợp lệ.")
                else:
                    # Chuẩn hóa mã lớp
                    safe_code = safe_filename(new_class_code_input)
                    if safe_code == "":
                        st.error("Mã lớp không hợp lệ. Vui lòng sử dụng chữ hoặc số.")
                    else:
                        # Cảnh báo nếu mã lớp được điều chỉnh khác với input ban đầu
                        if safe_code != new_class_code_input:
                            st.info(f"Mã lớp đã được chuyển thành **{safe_code}** để phù hợp.")
                        # Kiểm tra lớp đã tồn tại chưa
                        info_path = os.path.join(DATA_DIR, f"{safe_code}_info.json")
                        if os.path.exists(info_path):
                            st.error(f"⚠️ Lớp \"{safe_code}\" đã tồn tại. Hãy chọn mã lớp khác.")
                        else:
                            # Tạo file thông tin lớp mới
                            class_info = {
                                "class_code": safe_code,
                                "students": [],
                                "topics": []
                            }
                            with open(info_path, "w", encoding="utf-8") as f:
                                json.dump(class_info, f, ensure_ascii=False, indent=2)
                            st.success(f"✅ Đã tạo lớp mới với mã: **{safe_code}**")
                            # Cập nhật danh sách lớp
                            class_codes.append(safe_code)
                            class_codes.sort()
                            # Lưu mã lớp vừa tạo vào session_state để chọn mặc định
                            st.session_state["selected_class"] = safe_code

            # Nếu đã có ít nhất một lớp, cho phép chọn lớp để quản lý
            if class_codes:
                st.subheader("Quản lý lớp học")
                # Chọn lớp hiện tại
                default_class = None
                if "selected_class" in st.session_state and st.session_state["selected_class"] in class_codes:
                    default_class = st.session_state["selected_class"]
                selected_class = st.selectbox("Chọn lớp:", class_codes, index= class_codes.index(default_class) if default_class else 0)
                # Lưu lựa chọn lớp vào session_state
                st.session_state["selected_class"] = selected_class

                # Hiển thị thông tin lớp được chọn
                info_path = os.path.join(DATA_DIR, f"{selected_class}_info.json")
                if os.path.exists(info_path):
                    with open(info_path, "r", encoding="utf-8") as f:
                        class_info = json.load(f)
                    students = class_info.get("students", [])
                    topics = class_info.get("topics", [])
                    st.write(f"**Mã lớp:** {selected_class}")
                    st.write(f"**Số học sinh:** {len(students)} - " + (", ".join(students) if students else "Chưa có"))
                    st.write(f"**Số chủ đề kiến thức:** {len(topics)} - " + (", ".join(t['name'] for t in topics) if topics else "Chưa có"))
                else:
                    st.warning("Không tìm thấy thông tin lớp đã chọn.")

                # Thêm học sinh vào lớp
                st.subheader("Thêm học sinh")
                new_student = st.text_input("Nhập tên học sinh cần thêm:")
                add_student_btn = st.button("Thêm vào lớp")
                if add_student_btn:
                    if not new_student:
                        st.warning("Vui lòng nhập tên học sinh.")
                    else:
                        # Đọc thông tin lớp, thêm học sinh, lưu lại
                        with open(info_path, "r", encoding="utf-8") as f:
                            class_info = json.load(f)
                        # Kiểm tra trùng tên (không bắt buộc duy nhất nhưng tránh thêm 2 lần)
                        if new_student in class_info.get("students", []):
                            st.warning(f"Học sinh \"{new_student}\" đã có trong lớp.")
                        else:
                            class_info.setdefault("students", []).append(new_student)
                            with open(info_path, "w", encoding="utf-8") as f:
                                json.dump(class_info, f, ensure_ascii=False, indent=2)
                            st.success(f"✅ Đã thêm học sinh **{new_student}** vào lớp **{selected_class}**")

                # Upload tài liệu để tạo chủ đề mới
                st.subheader("Tạo chủ đề kiến thức")
                topic_name_input = st.text_input("Nhập tên chủ đề:")
                uploaded_file = st.file_uploader("Tải lên tài liệu cho chủ đề:", type=["pdf", "docx", "txt"])
                create_topic_btn = st.button("Tạo chủ đề")
                if create_topic_btn:
                    if not topic_name_input:
                        st.warning("Vui lòng nhập tên chủ đề.")
                    elif not uploaded_file:
                        st.warning("Vui lòng chọn một tệp tài liệu.")
                    else:
                        # Đảm bảo đã có thư mục data, tạo nếu chưa
                        if not os.path.exists(DATA_DIR):
                            os.makedirs(DATA_DIR)
                        # Đọc nội dung tài liệu
                        try:
                            document_text = read_document(uploaded_file)
                        except Exception as e:
                            st.error(f"❌ Lỗi: Không thể đọc file. Chi tiết: {e}")
                            document_text = None
                        if document_text is None or document_text.strip() == "":
                            st.error("❌ Không thể lấy nội dung từ tài liệu đã upload.")
                        else:
                            # Chia nhỏ tài liệu thành các đoạn (chunks)
                            chunks = split_into_chunks(document_text)
                            if not chunks or len(chunks) == 0:
                                st.error("❌ Tài liệu quá ngắn hoặc không thể tạo đoạn văn bản.")
                            else:
                                # Nhúng vector cho các đoạn văn bản (sử dụng hàm embed_texts)
                                with st.spinner("Đang xử lý và lưu trữ tri thức..."):
                                    embeddings = embed_texts(chunks)
                                    # Chuyển embeddings sang numpy array (nếu cần)
                                    embeddings = np.array(embeddings)
                                    # Tạo tên tập tin an toàn cho chủ đề
                                    safe_topic = safe_filename(topic_name_input)
                                    if safe_topic == "":
                                        safe_topic = "topic"
                                    # Kiểm tra chủ đề đã tồn tại trong lớp chưa (dựa trên tên file hoặc tên chủ đề)
                                    topic_exists = False
                                    for t in class_info.get("topics", []):
                                        if t["file"] == safe_topic or t["name"] == topic_name_input:
                                            topic_exists = True
                                            break
                                    if topic_exists:
                                        st.error("⚠️ Chủ đề này đã tồn tại. Vui lòng chọn tên khác.")
                                    else:
                                        # Lưu kiến thức (chunks và embeddings) ra file JSON + NPY
                                        save_knowledge(selected_class, safe_topic, chunks, embeddings)
                                        # Cập nhật thông tin chủ đề vào file info của lớp
                                        class_info.setdefault("topics", []).append({"name": topic_name_input, "file": safe_topic})
                                        with open(info_path, "w", encoding="utf-8") as f:
                                            json.dump(class_info, f, ensure_ascii=False, indent=2)
                                        st.success(f"✅ Đã tạo chủ đề **{topic_name_input}** cho lớp **{selected_class}**")
        else:
            st.sidebar.error("❌ Mã PIN không đúng. Vui lòng thử lại.")

    else:
        # Chưa nhập PIN, hiển thị hướng dẫn
        st.info("Vui lòng nhập mã PIN giáo viên ở thanh bên để sử dụng các chức năng quản lý lớp học.")

# Chế độ Học sinh
elif mode == "Học sinh":
    st.header("🎓 Chế độ Học sinh")
    # Nhập mã lớp để tham gia
    class_code = st.text_input("Nhập mã lớp:")
    join_btn = st.button("Vào lớp")
    if join_btn:
        if not class_code:
            st.warning("Vui lòng nhập mã lớp.")
        else:
            # Chuẩn hóa mã lớp giống như khi tạo (đảm bảo khớp tên file)
            class_code_safe = safe_filename(class_code)
            info_path = os.path.join(DATA_DIR, f"{class_code_safe}_info.json")
            if not os.path.exists(info_path):
                st.error("❌ Mã lớp không hợp lệ hoặc lớp chưa được tạo.")
            else:
                # Lưu mã lớp đã chọn vào session state để giữ trạng thái đăng nhập lớp
                st.session_state["current_class"] = class_code_safe
                st.session_state["current_topic"] = None
                st.success(f"✅ Đã vào lớp **{class_code_safe}**. Bây giờ bạn có thể chọn chủ đề và đặt câu hỏi.")
    # Nếu học sinh đã vào lớp (mã lớp hợp lệ được lưu)
    if "current_class" in st.session_state:
        class_code = st.session_state["current_class"]
        info_path = os.path.join(DATA_DIR, f"{class_code}_info.json")
        # Đọc thông tin lớp để lấy danh sách chủ đề
        with open(info_path, "r", encoding="utf-8") as f:
            class_info = json.load(f)
        topics = class_info.get("topics", [])
        if not topics:
            st.warning("Lớp này hiện chưa có chủ đề kiến thức nào. Vui lòng quay lại sau.")
        else:
            # Chọn chủ đề muốn hỏi
            topic_names = [t["name"] for t in topics]
            # Tìm chỉ số chủ đề đã chọn trước đó (nếu có) để đặt mặc định
            default_topic_idx = 0
            if st.session_state.get("current_topic") is not None:
                for idx, t in enumerate(topics):
                    if t["file"] == st.session_state["current_topic"]:
                        default_topic_idx = idx
                        break
            selected_topic_name = st.selectbox("Chọn chủ đề:", topic_names, index=default_topic_idx)
            # Xác định mã chủ đề (tên file an toàn) tương ứng
            topic_file = None
            for t in topics:
                if t["name"] == selected_topic_name:
                    topic_file = t["file"]
                    break
            if topic_file is None:
                st.error("Không tìm thấy chủ đề đã chọn.")
            else:
                # Cập nhật chủ đề hiện tại vào session (file name) nếu thay đổi
                if st.session_state.get("current_topic") != topic_file:
                    st.session_state["current_topic"] = topic_file
                    # Xóa câu hỏi và câu trả lời trước đó nếu có (khi đổi chủ đề)
                    if "last_answer" in st.session_state:
                        del st.session_state["last_answer"]
                # Nhập câu hỏi
                question = st.text_input("Đặt câu hỏi của bạn:")
                ask_btn = st.button("Hỏi")
                if ask_btn:
                    if not question:
                        st.warning("Vui lòng nhập câu hỏi.")
                    else:
                        # Tải tri thức của chủ đề (chunks và embeddings)
                        chunks, embeddings = load_knowledge(class_code, topic_file)
                        embeddings = np.array(embeddings)
                        # Tính vector embedding cho câu hỏi
                        question_embedding = embed_texts([question])
                        question_embedding = np.array(question_embedding[0] if isinstance(question_embedding, list) else question_embedding)
                        # Tính độ tương đồng cosine giữa câu hỏi và các đoạn kiến thức
                        # Sử dụng công thức: cos_sim = (A·B) / (||A||*||B||)
                        q_norm = np.linalg.norm(question_embedding)
                        emb_norms = np.linalg.norm(embeddings, axis=1)
                        # Tránh chia cho 0
                        denom = (emb_norms * q_norm + 1e-8)
                        cosine_similarities = np.dot(embeddings, question_embedding) / denom
                        # Lấy chỉ số của một số đoạn văn bản liên quan nhất (ví dụ top 3)
                        top_k = 3
                        if len(cosine_similarities) < top_k:
                            top_k = len(cosine_similarities)
                        top_indices = cosine_similarities.argsort()[::-1][:top_k]
                        relevant_chunks = [chunks[i] for i in top_indices]
                        # Gọi hàm sinh câu trả lời từ AI với ngữ cảnh
                        with st.spinner("Đang tìm câu trả lời..."):
                            answer = generate_answer(question, relevant_chunks)
                        # Hiển thị câu trả lời
                        st.write("**Trợ lý:** " + str(answer))
                        # Lưu câu trả lời vào session (có thể dùng nếu muốn hiển thị lại)
                        st.session_state["last_answer"] = str(answer)
    else:
        # Chưa nhập hoặc xác nhận mã lớp
        st.info("Hãy nhập mã lớp và nhấn 'Vào lớp' để bắt đầu.")
