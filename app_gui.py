import sys
import os
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QLineEdit, QLabel
)
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from milvus_utilis import save_to_milvus 
from embedding import split_into_chunks
from rag_chain import ask_llm_with_context
from api_embedding import delete_file

class TxtFolderWatcher(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        if event.src_path.endswith(".txt"):
            self.callback(event.src_path)

    def on_created(self, event):
        if event.src_path.endswith(".txt"):
            self.callback(event.src_path)


class AIReaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.watch_folder = None
        self.observer = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('🧠 AI Trợ lý đọc văn bản (.txt)')
        self.setGeometry(200, 200, 600, 500)

        layout = QVBoxLayout()

        # Nút chọn thư mục
        self.folder_btn = QPushButton("📁 Chọn thư mục chứa .txt")
        self.folder_btn.clicked.connect(self.choose_folder)
        layout.addWidget(self.folder_btn)

        # Hiển thị thư mục đã chọn
        self.folder_label = QLabel("📂 Chưa chọn thư mục")
        layout.addWidget(self.folder_label)

        # Ô nhập câu hỏi
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("✍️ Nhập câu hỏi của bạn...")
        layout.addWidget(self.query_input)

        # Nút hỏi AI
        self.ask_btn = QPushButton("🤖 Hỏi AI")
        self.ask_btn.clicked.connect(self.ask_ai)
        layout.addWidget(self.ask_btn)

        # Vùng hiển thị kết quả
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        layout.addWidget(self.result_area)

        self.setLayout(layout)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục")
        if folder:
            self.watch_folder = folder
            self.folder_label.setText(f"📂 Đã chọn: {folder}")
            self.process_txt_files(folder)
            self.start_watching_folder(folder)

    def process_txt_files(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                self.process_single_file(filepath)
        self.result_area.setText("✅ Đã xử lý và lưu tất cả file .txt vào Milvus.")

    def process_single_file(self, filepath):
        try:
            filename = os.path.basename(filepath)
            # Delete old entries first
            delete_file(filename)
            
            # Then process and save new content
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                chunks = split_into_chunks(text)
                save_to_milvus(chunks, filename)
                print(f"✅ Đã cập nhật: {filename}")
        except Exception as e:
            print(f"❌ Lỗi xử lý {filepath}: {str(e)}")

    def start_watching_folder(self, folder):
        if self.observer:
            self.observer.stop()

        event_handler = TxtFolderWatcher(self.process_single_file)
        self.observer = Observer()
        self.observer.schedule(event_handler, folder, recursive=False)

        observer_thread = threading.Thread(target=self.observer.start, daemon=True)
        observer_thread.start()

    def ask_ai(self):
        query = self.query_input.text().strip()
        if not query:
            self.result_area.setText("⚠️ Vui lòng nhập câu hỏi.")
            return
        try:
            answer = ask_llm_with_context(query)
            self.result_area.setText(answer)
        except Exception as e:
            self.result_area.setText(f"❌ Lỗi khi hỏi AI: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIReaderApp()
    window.show()
    sys.exit(app.exec_())
