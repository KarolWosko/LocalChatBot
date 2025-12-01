import sys
import traceback
import os

# Tworzenie folderów na pliki faiss oraz logi
os.makedirs("logs", exist_ok=True)
os.makedirs("faiss_data", exist_ok=True)

# Zapis błędów do pliku
sys.stdout = open("logs/log_output_app.txt", "w", encoding="utf-8")
sys.stderr = open("logs/log_errors_runtime.txt", "w", encoding="utf-8")

with open("logs/log_debug_startup.txt", "w", encoding="utf-8") as f:
    f.write("PYTHON STARTED\n")
    f.write(f"cwd = {os.getcwd()}\n")
    f.write(f"sys.frozen = {getattr(__import__('sys'), 'frozen', False)}\n")

import pickle
import faiss
import fitz
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QDialog, \
    QHBoxLayout, QPushButton, QTextEdit, QFileDialog, QLabel, QListWidget, QMessageBox, QComboBox
from PyQt6.QtCore import Qt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from llama_cpp import Llama
from docx import Document
import chardet

FAISS_INDEX_PATH = "faiss_data/faiss_index.bin"
TEXT_DATA_PATH = "faiss_data/text_chunks.pkl"

# Pobieramy ścieżkę do modeli
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS

EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
CHUNK_SIZE = 256

embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

def extract_text_from_pdf(file_path):
    """ Odczytuje tekst z pliku PDF """
    all_text = ""
    filename = os.path.basename(file_path)
    print(f"Przetwarzanie pliku: {filename}")
    doc = fitz.open(file_path)
    for page in doc:
        all_text += page.get_text("text") + "\n"
    return all_text

def extract_text_from_docx(file_path):
    """ Odczytuje tekst z pliku Word (.docx) """
    all_text = ""
    doc = Document(file_path)
    for paragraph in doc.paragraphs:
        all_text += paragraph.text + "\n"
    return all_text

def read_text_file_with_fallback(file_path):
    # Wczytujemy fragment pliku w trybie binarnym
    with open(file_path, "rb") as f:
        raw_data = f.read(4096)  # większa próbka = większa szansa na trafne wykrycie

    result = chardet.detect(raw_data)
    detected_encoding = result['encoding']
    confidence = result['confidence']

    log_msg = f"{os.path.basename(file_path)} - Wykryte kodowanie: {detected_encoding} (pewność: {confidence})\n"
    with open("logs/log_encoding_detected.txt", "a", encoding="utf-8") as log_file:
        log_file.write(log_msg)

    try:
        with open(file_path, "r", encoding=detected_encoding) as f:
            return f.read()
    except Exception as e:
        with open("logs/log_encoding_detected.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"BŁĄD podczas dekodowania {file_path}: {e}\n")
        raise UnicodeDecodeError(f"Nie udało się otworzyć pliku z użyciem {detected_encoding}")

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE):
    """ Dzieli tekst na fragmenty o określonej liczbie tokenów """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        if len(chunk_tokens) <= 256:
            chunk = tokenizer.decode(chunk_tokens)
            chunks.append(chunk)
    return chunks

def show_loading_dialog(parent, message="Przetwarzanie..."):
    """ Tworzy i wyświetla okno ładowania z paskiem postępu """
    loading_dialog = LoadingDialog(parent, message)
    loading_dialog.show()
    QApplication.processEvents()
    return loading_dialog

class LoadingDialog(QDialog):
    def __init__(self, parent=None, message="Przetwarzanie..."):
        super().__init__()
        self.setWindowTitle("Ładowanie")
        self.setModal(True)
        self.setFixedSize(300, 120)

        layout = QVBoxLayout()
        self.label = QLabel(message)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.setLayout(layout)

        if parent:
            self.center_on_parent(parent)

    def center_on_parent(self, parent):
        """ Umieszcza okno na środku głównego okna aplikacji """
        parent_rect = parent.geometry()
        self.move(parent_rect.center().x() - self.width() // 2,
                  parent_rect.center().y() - self.height() // 2)

class CustomTextEdit(QTextEdit):
    """ Pole do wpisania pytania, obsługujące Enter do wysyłania """

    def __init__(self, parent):
        super().__init__(parent)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            """ Jeśli naciśnięto Enter (bez Shift), wysyłamy pytanie """
            self.parent().handle_query()
        else:
            """ Standardowe zachowanie (przejście do nowej linii) """
            super().keyPressEvent(event)

class ChatbotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.show_history_button = None
        self.load_history_button = None
        self.save_history_button = None
        self.preview_button = None
        self.remove_all_button = None
        self.ask_button = None
        self.text_input = None
        self.response_area = None
        self.remove_button = None
        self.load_button = None
        self.file_list = None
        self.label = None
        self.chat_history = []
        self.model_combobox = None
        self.load_model_button = None
        self.model_name = "Nieznany"
        self.init_ui()
        self.loaded_files = []
        self.text_chunks = []
        self.index = None

        self.llm = None

        self.load_existing_data()

    def init_ui(self):
        self.setWindowTitle("Lokalny Chatbot do Analizy Dokumentów")
        self.setGeometry(100, 100, 1000, 400)

        main_layout = QHBoxLayout()

        """ Lewa kolumna """
        left_panel = QVBoxLayout()
        left_panel.setSpacing(4)
        left_panel.setContentsMargins(0, 0, 0, 0)

        self.model_combobox = QComboBox()
        self.model_combobox.addItems(["Wybierz model", "Mistral 7B", "Zephyr Beta"])
        left_panel.addWidget(QLabel("Wybór modelu językowego"))
        left_panel.addWidget(self.model_combobox)

        self.load_model_button = QPushButton("Załaduj model")
        self.load_model_button.clicked.connect(self.load_selected_model)
        left_panel.addWidget(self.load_model_button)

        self.label = QLabel("Wczytane dokumenty:")
        left_panel.addWidget(self.label)

        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(300)
        left_panel.addWidget(self.file_list)

        """ Środkowa kolumna """
        middle_panel = QVBoxLayout()
        middle_panel.setSpacing(2)
        middle_panel.setContentsMargins(0, 0, 0, 0)

        self.load_button = QPushButton("Wczytaj plik")
        self.load_button.setFixedHeight(30)
        self.load_button.clicked.connect(self.load_file)
        middle_panel.addWidget(self.load_button)

        self.preview_button = QPushButton("Podgląd pliku")
        self.preview_button.setFixedHeight(30)
        self.preview_button.clicked.connect(self.preview_file)
        middle_panel.addWidget(self.preview_button)

        self.remove_button = QPushButton("Usuń wybrany plik")
        self.remove_button.setFixedHeight(30)
        self.remove_button.clicked.connect(self.remove_file)
        middle_panel.addWidget(self.remove_button)

        self.remove_all_button = QPushButton("Usuń wszystkie pliki")
        self.remove_all_button.setFixedHeight(30)
        self.remove_all_button.clicked.connect(self.remove_all_files)
        middle_panel.addWidget(self.remove_all_button)

        self.show_history_button = QPushButton("Pokaż bieżącą rozmowę")
        self.show_history_button.setFixedHeight(30)
        self.show_history_button.clicked.connect(self.show_chat_history)
        middle_panel.addWidget(self.show_history_button)

        self.save_history_button = QPushButton("Zapisz historię rozmowy")
        self.save_history_button.setFixedHeight(30)
        self.save_history_button.clicked.connect(self.save_chat_history)
        middle_panel.addWidget(self.save_history_button)

        self.load_history_button = QPushButton("Wczytaj historię rozmowy")
        self.load_history_button.setFixedHeight(30)
        self.load_history_button.clicked.connect(self.load_chat_history)
        middle_panel.addWidget(self.load_history_button)

        """ Prawa kolumna """
        right_panel = QVBoxLayout()
        right_panel.setSpacing(4)
        right_panel.setContentsMargins(0, 0, 0, 0)

        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setMinimumHeight(300)
        right_panel.addWidget(self.response_area)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(4)
        input_layout.setContentsMargins(0, 0, 0, 0)

        self.text_input = CustomTextEdit(self)
        self.text_input.setPlaceholderText("Zadaj pytanie dotyczące dokumentów...")
        self.text_input.setFixedHeight(50)
        input_layout.addWidget(self.text_input)

        self.ask_button = QPushButton("Zapytaj")
        self.ask_button.setFixedHeight(30)
        self.ask_button.clicked.connect(self.handle_query)
        input_layout.addWidget(self.ask_button)

        right_panel.addLayout(input_layout)

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(middle_panel, 2)
        main_layout.addLayout(right_panel, 6)

        self.setLayout(main_layout)

    def load_selected_model(self):
        selected_model = self.model_combobox.currentText()

        if selected_model not in ["Mistral 7B", "Zephyr Beta"]:
            QMessageBox.warning(self, "Błąd", "Wybierz poprawny model.")
            return

        loading_dialog = show_loading_dialog(self, f"Ładowanie modelu: {selected_model}...")

        try:
            if selected_model == "Mistral 7B":
                model_path = os.path.join(BASE_DIR, "language_model", "mistral-7b-instruct-v0.2.Q2_K.gguf")
            elif selected_model == "Zephyr Beta":
                model_path = os.path.join(BASE_DIR, "language_model", "zephyr-7b-beta.Q2_K.gguf")
            else:
                raise ValueError("Nieznany model.")

            # Faktyczne ładowanie modelu bez subprocess
            QApplication.processEvents()
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_threads=4,
                    n_batch=256,
                    use_mmap=False,
                )
            except Exception as e:
                error_text = f"Nie udało się załadować modelu:\n{str(e)}"
                QMessageBox.critical(self, "Błąd ładowania modelu", error_text)
                self.llm = None
                return
            self.model_name = selected_model
            QMessageBox.information(self, "Sukces", f"Model {selected_model} został poprawnie załadowany.")

        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się załadować modelu: {e}")
        finally:
            loading_dialog.close()

    def handle_query(self):
        query = self.text_input.toPlainText().strip()

        if not self.llm:
            self.response_area.setText("Najpierw wybierz i załaduj model językowy!")
            return

        if query:
            loading_dialog = show_loading_dialog(self, "Generowanie odpowiedzi...")
            self.search_in_faiss(query)
            loading_dialog.close()
        else:
            self.response_area.setText("Wpisz pytanie przed wyszukiwaniem!")

    def load_existing_data(self):
        """ Wczytaj istniejące dane """
        if os.path.exists(TEXT_DATA_PATH):
            with open(TEXT_DATA_PATH, "rb") as f:
                self.text_chunks = pickle.load(f)

        if os.path.exists(FAISS_INDEX_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            self.index = None

    def load_file(self):
        """ Wczytaj plik PDF lub txt """
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, "Wybierz plik", "",
                                                       "Obsługiwane pliki (*.pdf *.txt *.docx)")
            if file_path:
                loading_dialog = show_loading_dialog(self, f"Wczytywanie dokumentu: {os.path.basename(file_path)}...")

                if file_path not in self.loaded_files:
                    self.loaded_files.append(file_path)
                    self.file_list.addItem(os.path.basename(file_path))

                    if file_path.endswith(".pdf"):
                        text = extract_text_from_pdf(file_path)
                    elif file_path.endswith(".txt"):
                        text = read_text_file_with_fallback(file_path)
                    elif file_path.endswith(".docx"):
                        text = extract_text_from_docx(file_path)

                    chunks = split_text_into_chunks(text)
                    self.text_chunks.extend(chunks)

                    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

                    if self.index is None:
                        dimension = embeddings.shape[1]
                        self.index = faiss.IndexFlatL2(dimension)
                    self.index.add(embeddings)
                    print(f"Aktualna liczba wektorów w FAISS: {self.index.ntotal}")

                    faiss.write_index(self.index, FAISS_INDEX_PATH)
                    with open(TEXT_DATA_PATH, "wb") as f:
                        pickle.dump(self.text_chunks, f)

                    print(f"Przetworzono i zapisano: {file_path}")

                loading_dialog.close()
        except Exception:
            with open("logs/log_error_file_load.txt", "w", encoding="utf-8") as log:
                log.write("Błąd podczas wczytywania pliku:\n")
                log.write(traceback.format_exc())

    def count_tokens(self, text):
        return len(self.llm.tokenize(text.encode("utf-8")))

    def search_in_faiss(self, query, top_k=1):
        """ Wyszukuje w FAISS najbardziej dopasowane fragmenty do zapytania
        użytkownika i generuje odpowiedź za pomocą modelu językowego,
        a następnie zapisuje historie rozmowy. """

        if self.index is None or not self.text_chunks:
            self.response_area.setText("Brak danych! Dodaj pliki, aby wyszukiwać odpowiedzi")
            return

        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        context = [self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]

        prompt = f"""<|system|>
                    Jesteś inteligentnym asystentem AI do analizy dokumentów.
                    Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów.
                    <|user|>
                    Poniżej znajdują się fragmenty z dokumentów:
                    {context}
                    Twoim zadaniem jest udzielenie jasnej, rzeczowej i precyzyjnej odpowiedzi
                    na pytanie użytkownika. Nie zgaduj. Jeśli odpowiedź nie wynika bezpośrednio z
                    kontekstu, napisz: "Nie znalazłem odpowiedzi w dostępnych fragmentach."
                    Pytanie użytkownika:
                    {query}
                    <|assistant|>"""

        prompt_token_count = self.count_tokens(prompt)
        n_ctx = 4096

        if prompt_token_count >= n_ctx:
            self.response_area.setText("Zbyt długi prompt! Skróć pytanie lub kontekst.")
            return

        max_response_tokens = max(256, n_ctx - prompt_token_count)
        response = self.llm(prompt, max_tokens=max_response_tokens, temperature=0.7, repeat_penalty=1.1)
        answer = response["choices"][0]["text"].strip()

        """ Zapis do historii rozmowy """
        self.chat_history.append(f"Model: {self.model_name}\nPytanie: {query} \nOdpowiedź: {answer} \n")

        self.response_area.setText(answer)

    def remove_file(self):
        """ Usuwa wybrany plik z listy i wywołuje funkcje do ponownego
        wygenerowania FAISS """
        selected_item = self.file_list.currentRow()
        if selected_item >= 0:
            removed_filename = self.file_list.item(selected_item).text()

            loading_dialog = show_loading_dialog(self, f"Usuwanie pliku: {removed_filename}...")

            self.file_list.takeItem(selected_item)

            removed_file_path = None
            for file_path in self.loaded_files:
                if os.path.basename(file_path) == removed_filename:
                    removed_file_path = file_path
                    break

            if removed_file_path:
                self.loaded_files.remove(removed_file_path)
                print(f"Usunięto plik: {removed_filename}")

            self.rebuild_faiss()
            loading_dialog.close()

    def remove_all_files(self):
        """ Usuwa wszystkie pliki z listy oraz resetuje FAISS """
        loading_dialog = show_loading_dialog(self, "Usuwanie wszystkich plików...")
        self.file_list.clear()
        self.loaded_files = []
        self.text_chunks = []

        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(TEXT_DATA_PATH):
            os.remove(TEXT_DATA_PATH)

        self.index = None
        loading_dialog.close()
        print("Wszystkie pliki usunięte. FAISS został zresetowany.")

    def rebuild_faiss(self):
        """ Przebudowuje dane FAISS po usunięciu pliku,
        uwzględniając tylko istniejące pliki. """
        self.text_chunks = []
        valid_files = []

        for file_path in self.loaded_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
                if file_path.endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)
                elif file_path.endswith(".txt"):
                    text = read_text_file_with_fallback(file_path)
                elif file_path.endswith(".docx"):
                    text = extract_text_from_docx(file_path)
                self.text_chunks.extend(split_text_into_chunks(text))
            else:
                print(f"Plik {file_path} nie istnieje, pomijam.")

        if valid_files:
            embeddings = embedding_model.encode(self.text_chunks, convert_to_numpy=True)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)

            faiss.write_index(self.index, FAISS_INDEX_PATH)
            with open(TEXT_DATA_PATH, "wb") as f:
                pickle.dump(self.text_chunks, f)

            self.loaded_files = valid_files
            print(f"FAISS przebudowane po usunięciu pliku. Liczba wektorów: {self.index.ntotal}")
        else:
            if os.path.exists(FAISS_INDEX_PATH):
                os.remove(FAISS_INDEX_PATH)
            if os.path.exists(TEXT_DATA_PATH):
                os.remove(TEXT_DATA_PATH)
            self.index = None
            self.text_chunks = []
            self.loaded_files = []
            print("Wszystkie pliki usunięte. FAISS został zresetowany.")

    def preview_file(self):
        """ Wyświetla podgląd treści wybranego pliku w oknie chatu """
        selected_item = self.file_list.currentRow()
        if selected_item >= 0:
            filename = self.file_list.item(selected_item).text()
            loading_dialog = show_loading_dialog(self, f"Ładowanie podglądu pliku: {filename}...")

            file_path = None
            for path in self.loaded_files:
                if os.path.basename(path) == filename:
                    file_path = path
                    break

            if not file_path or not os.path.exists(file_path):
                self.response_area.setText("Plik nie istnieje.")
                return

            if file_path.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith(".txt"):
                text = read_text_file_with_fallback(file_path)
            elif file_path.endswith(".docx"):
                text = extract_text_from_docx(file_path)
            else:
                self.response_area.setText("Nieobsługiwany format pliku.")
                return
            loading_dialog.close()
            self.response_area.setText(f"** Podgląd pliku: {filename} **\n\n{text[:2000]}...")
        else:
            self.response_area.setText("Wybierz plik do podglądu")

    def show_chat_history(self):
        """ Wyświetla bieżącą rozmowę w osobnym oknie """
        if not self.chat_history:
            QMessageBox.information(self, "Brak rozmowy", "Nie odbyto jeszcze żadnej rozmowy.")
            return

        history_dialog = QDialog(self)
        history_dialog.setWindowTitle("Bieżąca rozmowa")
        history_dialog.setModal(True)
        history_dialog.resize(600, 400)

        layout = QVBoxLayout()
        text_box = QTextEdit()
        text_box.setReadOnly(True)
        text_box.setText("".join(self.chat_history))
        layout.addWidget(text_box)

        close_button = QPushButton("Zamknij")
        close_button.clicked.connect(history_dialog.close)
        layout.addWidget(close_button)

        history_dialog.setLayout(layout)
        history_dialog.exec()

    def save_chat_history(self):
        """ Zapisuje historię rozmowy do pliku tekstowego """
        if not self.chat_history:
            QMessageBox.information(self, "Zapis historii", "Brak historii rozmowy do zapisania.")
            return

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Zapisz historię", "", "Pliki tekstowe (*.txt)")

        if file_path:
            loading_dialog = show_loading_dialog(self, "Zapisywanie historii...")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.chat_history))
            loading_dialog.close()
            QMessageBox.information(self, "Zapis historii", "Historia rozmowy została zapisana.")

    def load_chat_history(self):
        """Wczytuje historię rozmowy z pliku."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Otwórz historię rozmowy", "", "Pliki tekstowe (*.txt)")

        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                self.chat_history = f.readlines()
            self.response_area.setText("Wczytana historia rozmowy:\n\n" + "".join(self.chat_history))

    def closeEvent(self, event):
        """ Sprawdza przed zamknięciem aplikacji, czy FAISS nie został zresetowany
        i pyta użytkownika, czy chce usunąć wczytane dane """
        if self.index is not None and self.loaded_files:
            close_box = QMessageBox(self)
            close_box.setIcon(QMessageBox.Icon.Question)
            close_box.setWindowTitle("Zamknięcie aplikacji")
            close_box.setText("Czy chcesz usunąć wczytane dane?")

            tak_button = close_box.addButton("Tak", QMessageBox.ButtonRole.YesRole)
            nie_button = close_box.addButton("Nie", QMessageBox.ButtonRole.NoRole)

            close_box.setDefaultButton(nie_button)
            close_box.exec()

            if close_box.clickedButton() == tak_button:
                self.remove_all_files()
                print("Dane zostały usunięte przed zamknięciem.")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatbotApp()
    window.show()
    sys.exit(app.exec())