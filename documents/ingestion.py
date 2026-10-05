from pathlib import Path
import pandas as pd
from docx import Document
from PIL import Image
import pytesseract
from pypdf import PdfReader

def read_excel(path: Path) -> str:
    try:
        df = pd.read_excel(path)
        rows = []
        for _, r in df.iterrows():
            rows.append(' '.join([str(v) for v in r.tolist() if pd.notna(v)]))
        return '\n'.join(rows)
    except Exception as e:
        return f"[excel error] {e}"

def read_word(path: Path) -> str:
    try:
        doc = Document(path)
        return '\n'.join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"[word error] {e}"

def read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        texts = []
        for p in reader.pages:
            try:
                texts.append(p.extract_text() or "")
            except Exception:
                texts.append("")
        return '\n'.join(texts)
    except Exception as e:
        return f"[pdf error] {e}"

def read_image_ocr(path: Path) -> str:
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        return f"[ocr error] {e}"

def ingest_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        return read_excel(path)
    if suffix in ('.docx',):
        return read_word(path)
    if suffix in ('.pdf',):
        return read_pdf(path)
    if suffix in ('.png', '.jpg', '.jpeg', '.tiff'):
        return read_image_ocr(path)
    # Fallback: try to read as text
    try:
        return path.read_text(encoding='utf-8')
    except Exception as e:
        return f"[read error] {e}"


class DataIngestionPipeline:
    SUPPORTED_EXTENSIONS = {'.xlsx', '.xls', '.docx', '.pdf', '.png', '.jpg', '.jpeg', '.tiff'}

    def __init__(self, source_dir: Path):
        self.source_dir = source_dir

    def list_files(self):
        if not self.source_dir.exists():
            return []
        return [p for p in sorted(self.source_dir.iterdir()) if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS]

    def ingest(self):
        documents = []
        for path in self.list_files():
            text = ingest_file(path)
            documents.append({'path': path, 'text': text})
        return documents
