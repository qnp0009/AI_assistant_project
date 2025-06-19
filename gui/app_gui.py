import sys
import os
import threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QLineEdit, QLabel
)
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from core.milvus_utilis import save_to_milvus, delete_file
from core.embedding import split_into_chunks
from core.rag_chain import ask_question_smart_with_toolcall
import fitz 

# Watches a folder for new or changed .txt/.pdf files
class TxtFolderWatcher(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        # Ensure src_path is str for type checking
        src_path = str(event.src_path)
        if src_path.endswith('.txt') or src_path.endswith('.pdf'):
            self.callback(src_path)

    def on_created(self, event):
        # Ensure src_path is str for type checking
        src_path = str(event.src_path)
        if src_path.endswith('.txt') or src_path.endswith('.pdf'):
            self.callback(src_path)

# Main application window
class AIReaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.watch_folder = None
        self.observer = None
        self.init_ui()

    def init_ui(self):
        # Set up the UI layout and widgets
        self.setWindowTitle('🧠 AI Document Assistant (.txt, .pdf)')
        self.setGeometry(200, 200, 600, 500)

        layout = QVBoxLayout()

        # Folder selection button
        self.folder_btn = QPushButton("📁 Select folder with .txt, .pdf files")
        self.folder_btn.clicked.connect(self.choose_folder)
        layout.addWidget(self.folder_btn)

        # Display selected folder
        self.folder_label = QLabel("📂 No folder selected")
        layout.addWidget(self.folder_label)

        # Query input box
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("✍️ Enter your question...")
        layout.addWidget(self.query_input)

        # Ask AI button
        self.ask_btn = QPushButton("🤖 Ask AI")
        self.ask_btn.clicked.connect(self.ask_ai)
        layout.addWidget(self.ask_btn)

        # Delete file button
        self.delete_btn = QPushButton("🗑️ Delete file")
        self.delete_btn.clicked.connect(self.delete_file)
        layout.addWidget(self.delete_btn)

        # Result display area
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        layout.addWidget(self.result_area)

        self.setLayout(layout)

    def choose_folder(self):
        # Let user select a folder and process its files
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            self.watch_folder = folder
            self.folder_label.setText(f"📂 Selected: {folder}")
            self.process_supported_files(folder)
            self.start_watching_folder(folder)

    def process_supported_files(self, folder_path):
        # Process all .txt and .pdf files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt') or filename.endswith('.pdf'):
                filepath = os.path.join(folder_path, filename)
                self.process_single_file(filepath)
        self.result_area.setText("✅ All .txt and .pdf files have been processed and saved to Milvus.")

    def process_single_file(self, filepath):
        # Read, chunk, and save a single file
        try:
            filename = os.path.basename(filepath)
            delete_file(filename)  # Delete old data in Milvus

            if filepath.endswith('.txt'):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            elif filepath.endswith('.pdf'):
                text = self.read_pdf_text(filepath)
            else:
                return 

            chunks = split_into_chunks(text)
            save_to_milvus(chunks, filename)
            print(f"✅ Updated: {filename}")
        except Exception as e:
            print(f"❌ Error processing {filepath}: {str(e)}")

    def start_watching_folder(self, folder):
        # Start watching the folder for new or changed files
        if self.observer:
            self.observer.stop()

        event_handler = TxtFolderWatcher(self.process_single_file)
        self.observer = Observer()
        self.observer.schedule(event_handler, folder, recursive=False)

        observer_thread = threading.Thread(target=self.observer.start, daemon=True)
        observer_thread.start()

    def read_pdf_text(self, filepath):
        # Extract text from a PDF file
        text = ""
        try:
            doc = fitz.open(filepath)
            for page in doc:
                # Use get_text (PyMuPDF >= 1.18.0)
                text += page.get_text()
            doc.close()
            print(f"✅ Successfully read PDF: {filepath}")
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
        return text

    def ask_ai(self):
        # Send user query to AI and display the answer
        query = self.query_input.text().strip()
        if not query:
            self.result_area.setText("⚠️ Please enter a question.")
            return
        try:
            answer = ask_question_smart_with_toolcall(query)
            self.result_area.setText(answer)
        except Exception as e:
            self.result_area.setText(f"❌ Error asking AI: {str(e)}")

    def delete_file(self):
        # Let user enter the filename and click the button to delete from Milvus
        if not self.watch_folder:
            self.result_area.setText("⚠️ Please select a folder first.")
            return
            
        file_dialog = QFileDialog()
        file_dialog.setDirectory(self.watch_folder)
        file_dialog.setNameFilter("Text files (*.txt);;PDF files (*.pdf)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                filepath = selected_files[0]
                filename = os.path.basename(filepath)
                try:
                    result = delete_file(filename)
                    self.result_area.setText(result["message"])
                except Exception as e:
                    self.result_area.setText(f"❌ Error deleting file: {str(e)}")

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIReaderApp()
    window.show()
    sys.exit(app.exec_())
