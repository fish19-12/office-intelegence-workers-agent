"""Document ingestion, parsing, normalization, and chunking pipeline.

This module preserves the existing DocumentManager API while upgrading the
internal document-processing flow to a modular, validation-first pipeline.

Only the document loading, parsing, normalization, and chunking concerns are
handled here; retrieval, embeddings, and agent logic remain intentionally out of
scope.
"""

from __future__ import annotations

import logging
import hashlib
import mimetypes
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

try:  # pragma: no cover
    import magic
except Exception:  # pragma: no cover
    magic = None

try:  # pragma: no cover
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:  # pragma: no cover
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:  # pragma: no cover
    import fitz
except Exception:  # pragma: no cover
    fitz = None

try:  # pragma: no cover
    from docx import Document as DocxDocument
except Exception:  # pragma: no cover
    DocxDocument = None

try:  # pragma: no cover
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:  # pragma: no cover
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:  # pragma: no cover
    from langdetect import detect as langdetect_detect
except Exception:  # pragma: no cover
    langdetect_detect = None

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Base document-processing exception."""

    def __init__(
        self,
        message: str,
        *,
        document: Optional[str] = None,
        stage: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        self.message = message
        self.document = document
        self.stage = stage
        self.cause = cause
        super().__init__(message)


class UnsupportedFormatError(DocumentProcessingError):
    """Raised when the file extension or MIME type is not supported."""


class InvalidFileError(DocumentProcessingError):
    """Raised when the file is invalid or malformed."""


class FileTooLargeError(DocumentProcessingError):
    """Raised when the file exceeds the configured size limit."""


class ParsingFailureError(DocumentProcessingError):
    """Raised when a parser cannot read the content successfully."""


class OCRFailureError(DocumentProcessingError):
    """Raised when OCR extraction fails."""


class NormalizationFailureError(DocumentProcessingError):
    """Raised when the parsed content cannot be normalized."""


class ChunkingFailureError(DocumentProcessingError):
    """Raised when chunk generation fails."""


class InvalidChunkError(DocumentProcessingError):
    """Raised when a chunk is invalid."""


DEFAULT_SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".csv",
    ".xlsx",
    ".xls",
    ".docx",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
]


@dataclass
class LoadedDocument:
    """File validation and loading result."""

    path: Path
    extension: str
    mime_type: str
    size_bytes: int
    detected_format: str
    source_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingConfig:
    """Typed document processing configuration object."""

    parser_settings: Dict[str, Any] = field(default_factory=dict)
    ocr_settings: Dict[str, Any] = field(default_factory=lambda: {"languages": ["eng"]})
    language_settings: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "min_chars": 25})
    chunking_strategy: str = "balanced"
    chunking_profile: str = "balanced"
    chunk_size: int = 1600
    overlap: int = 200
    min_chunk_size: int = 80
    max_chunk_size: int = 3000
    token_limit: Optional[int] = None
    tokenizer_name: Optional[str] = None
    semantic_threshold: float = 0.8
    table_row_group_size: int = 25
    parent_child_mode: bool = False
    deduplication_behavior: str = "safe"
    validation_behavior: str = "strict"
    language_detection: bool = True
    ocr_languages: Optional[List[str]] = None
    supported_extensions: List[str] = field(default_factory=lambda: DEFAULT_SUPPORTED_EXTENSIONS.copy())

    def __post_init__(self) -> None:
        if self.ocr_languages is None:
            self.ocr_languages = self.ocr_settings.get("languages", ["eng"])
        self.chunking_strategy = (self.chunking_strategy or "balanced").lower()
        self.chunking_profile = (self.chunking_profile or "balanced").lower()
        if self.chunk_size <= 0:
            self.chunk_size = 1600
        if self.overlap < 0:
            self.overlap = 0
        if self.overlap >= self.chunk_size:
            self.overlap = max(0, self.chunk_size // 2)
        if self.min_chunk_size <= 0:
            self.min_chunk_size = 50
        if self.max_chunk_size <= 0 or self.max_chunk_size < self.chunk_size:
            self.max_chunk_size = max(self.chunk_size, 3000)


@dataclass
class Chunk:
    """Normalized chunk output."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FinalizedChunk:
    """Embedding-ready text plus the finalized chunk metadata."""

    content: str
    embedding_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkFinalizationLayer:
    """Validate, normalize, deduplicate, and prepare chunks for embedding."""

    def finalize(
        self,
        chunks: Sequence[Chunk],
        *,
        document: Optional[Dict[str, Any]] = None,
        config: Optional[ProcessingConfig] = None,
    ) -> List[FinalizedChunk]:
        config = config or ProcessingConfig()
        document = document or {}
        finalized: List[FinalizedChunk] = []
        seen_hashes: set[str] = set()

        for source_index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise InvalidChunkError(
                    "Chunk finalization requires Chunk instances.",
                    document=document.get("document_id"),
                    stage="chunk_validation",
                )

            normalized_content = self._normalize_content(chunk.content)
            if not normalized_content:
                raise InvalidChunkError(
                    "Chunk content cannot be empty after normalization.",
                    document=document.get("document_id"),
                    stage="chunk_validation",
                )
            if len(normalized_content) > config.max_chunk_size:
                raise InvalidChunkError(
                    f"Chunk exceeds the configured maximum size of {config.max_chunk_size} characters.",
                    document=document.get("document_id"),
                    stage="chunk_validation",
                )

            content_hash = hashlib.sha256(normalized_content.casefold().encode("utf-8")).hexdigest()
            if config.deduplication_behavior != "disabled" and content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            metadata = dict(chunk.metadata or {})
            embedding_text = self._prepare_embedding_text(normalized_content, metadata)
            metadata.update(
                {
                    "chunk_index": len(finalized),
                    "source_chunk_index": source_index,
                    "chunk_id": metadata.get("chunk_id") or f"{document.get('document_id', 'document')}:chunk:{len(finalized)}",
                    "document_id": metadata.get("document_id") or document.get("document_id"),
                    "content_hash": content_hash,
                    "embedding_text": embedding_text,
                    "finalization_status": "ready",
                    "finalization_version": 1,
                    "character_count": len(normalized_content),
                    "token_count": estimate_token_count(embedding_text, config.tokenizer_name),
                    "finalized_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            finalized.append(FinalizedChunk(normalized_content, embedding_text, metadata))

        return finalized

    def _normalize_content(self, content: Any) -> str:
        if not isinstance(content, str):
            raise InvalidChunkError("Chunk content must be text.", stage="chunk_normalization")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in content.replace("\x00", "").splitlines()]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    def _prepare_embedding_text(self, content: str, metadata: Dict[str, Any]) -> str:
        heading_path = metadata.get("heading_path") or []
        heading = " > ".join(str(item).strip() for item in heading_path if str(item).strip())
        if heading and not content.casefold().startswith(heading.casefold()):
            return f"{heading}\n\n{content}"
        return content


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def detect_language(text: str, *, min_chars: int = 25, enabled: bool = True) -> str:
    if not enabled or not text or len(text.strip()) < min_chars:
        return "unknown"
    try:
        if langdetect_detect is None:
            return "unknown"
        result = str(langdetect_detect(text[:500]))
        return result if result else "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def estimate_token_count(text: str, model_name: Optional[str] = None) -> int:
    if not text:
        return 0
    if model_name and model_name.lower().startswith("gpt"):
        words = re.findall(r"\S+", text)
        return max(1, int(len(words) * 1.3))
    return max(1, len(re.findall(r"\S+", text)))


def build_document_hierarchy(document: Dict[str, Any]) -> Dict[str, Any]:
    """Attach explicit page and section relationships to a normalized document."""
    pages = document.get("pages") or []
    sections: List[Dict[str, Any]] = []
    hierarchy_pages: List[Dict[str, Any]] = []

    for page in pages:
        page_number = page.get("page_number")
        section_by_path: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        page_sections: List[Dict[str, Any]] = []
        for block in page.get("blocks") or []:
            heading_path = tuple(block.get("heading_path") or ())
            section_key = heading_path or ("Unsectioned",)
            section = section_by_path.get(section_key)
            if section is None:
                section_id = f"{document.get('document_id', 'document')}:page:{page_number}:section:{len(page_sections)}"
                section = {
                    "section_id": section_id,
                    "title": section_key[-1],
                    "heading_path": list(heading_path),
                    "page_number": page_number,
                    "blocks": [],
                }
                section_by_path[section_key] = section
                page_sections.append(section)
                sections.append(section)

            block["document_id"] = document.get("document_id")
            block["page_number"] = page_number
            block["section_id"] = section["section_id"]
            section["blocks"].append(block)

        page["sections"] = page_sections
        hierarchy_pages.append({"page_number": page_number, "sections": page_sections})

    document["sections"] = sections
    document["hierarchy"] = {"document_id": document.get("document_id"), "pages": hierarchy_pages}
    return document


class DocumentLoader:
    """Safe file validation and loading with MIME detection."""

    def __init__(
        self,
        max_file_size_bytes: int = 50 * 1024 * 1024,
        allowed_roots: Optional[Sequence[Union[str, Path]]] = None,
    ) -> None:
        self.max_file_size_bytes = max_file_size_bytes
        default_roots = [Path.cwd(), Path(tempfile.gettempdir())]
        self.allowed_roots = [Path(root).resolve() for root in (allowed_roots or default_roots)]

    def _validate_path(self, file_path: Union[str, Path]) -> Path:
        raw_path = Path(file_path)
        if ".." in raw_path.parts:
            raise ValueError("Path traversal is not allowed.")

        resolved = raw_path.resolve(strict=False)
        allowed = any(resolved.is_relative_to(root) for root in self.allowed_roots)
        if not allowed:
            raise ValueError(f"Path is outside the allowed filesystem roots: {file_path}")

        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not resolved.is_file():
            raise InvalidFileError("The path does not point to a regular file.", document=str(file_path), stage="file_validation")
        return resolved

    def _mime_type_for(self, path: Path) -> str:
        try:
            if magic is not None:
                detected = magic.from_file(str(path), mime=True)
                if detected:
                    return str(detected)
        except Exception:  # pragma: no cover
            pass
        guessed, _ = mimetypes.guess_type(str(path))
        return guessed or "application/octet-stream"

    def load(self, file_path: Union[str, Path]) -> LoadedDocument:
        resolved = self._validate_path(file_path)
        suffix = resolved.suffix.lower()
        if not suffix:
            raise UnsupportedFormatError(
                "The file has no extension and cannot be identified safely.",
                document=str(resolved),
                stage="file_validation",
            )

        size = resolved.stat().st_size
        if size <= 0:
            raise InvalidFileError("The file is empty.", document=str(resolved), stage="file_validation")
        if size > self.max_file_size_bytes:
            raise FileTooLargeError(
                f"File exceeds the configured maximum size of {self.max_file_size_bytes} bytes.",
                document=str(resolved),
                stage="file_validation",
            )

        mime_type = self._mime_type_for(resolved)
        return LoadedDocument(
            path=resolved,
            extension=suffix,
            mime_type=mime_type,
            size_bytes=size,
            detected_format=suffix,
            source_metadata={"source_path": str(resolved), "mime_type": mime_type, "size_bytes": size},
        )


class BaseDocumentParser:
    """Parser interface used by the router."""

    supported_extensions: Tuple[str, ...] = ()

    def parse(self, loaded_document: LoadedDocument, config: Optional[ProcessingConfig] = None) -> Dict[str, Any]:
        raise NotImplementedError


class PlainTextParser(BaseDocumentParser):
    supported_extensions = (".txt", ".md", ".html", ".htm")

    def parse(self, loaded_document: LoadedDocument, config: Optional[ProcessingConfig] = None) -> Dict[str, Any]:
        config = config or ProcessingConfig()
        try:
            text = loaded_document.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = loaded_document.path.read_bytes().decode("utf-8", errors="replace")

        blocks: List[Dict[str, Any]] = []
        for index, chunk in enumerate(re.split(r"\n{2,}", text.strip())):
            cleaned = safe_text(chunk)
            if not cleaned:
                continue
            blocks.append(
                {
                    "block_type": "paragraph",
                    "text": cleaned,
                    "heading_path": [],
                    "heading_level": None,
                    "page_number": 1,
                    "source_location": {"page": 1, "position": index},
                    "metadata": {"extraction_method": "text"},
                }
            )

        return {
            "document_id": str(uuid4()),
            "title": loaded_document.path.stem,
            "document_type": "text",
            "language": detect_language(
                text,
                min_chars=config.language_settings.get("min_chars", 25),
                enabled=config.language_detection,
            ),
            "pages": [{"page_number": 1, "blocks": blocks, "metadata": {}}],
            "blocks": blocks,
            "sections": [],
            "metadata": {
                "source_path": str(loaded_document.path),
                "file_type": loaded_document.extension,
                "mime_type": loaded_document.mime_type,
            },
        }


class CSVParser(BaseDocumentParser):
    supported_extensions = (".csv",)

    def parse(self, loaded_document: LoadedDocument, config: Optional[ProcessingConfig] = None) -> Dict[str, Any]:
        config = config or ProcessingConfig()
        if pd is None:
            raise ParsingFailureError("pandas is required for CSV parsing.", document=str(loaded_document.path), stage="parsing")

        try:
            df = pd.read_csv(loaded_document.path)
        except Exception as exc:
            raise ParsingFailureError("Failed to read CSV file.", document=str(loaded_document.path), stage="parsing", cause=exc)

        block = {
            "block_type": "table",
            "text": df.to_string(index=False),
            "heading_path": [],
            "heading_level": None,
            "page_number": 1,
            "source_location": {"page": 1, "sheet": "Sheet1"},
            "metadata": {
                "worksheet": "Sheet1",
                "headers": [safe_text(col) for col in df.columns],
                "row_count": len(df.index),
                "column_count": len(df.columns),
                "extraction_method": "structured_csv",
            },
        }
        return {
            "document_id": str(uuid4()),
            "title": loaded_document.path.stem,
            "document_type": "csv",
            "language": detect_language(block["text"], min_chars=config.language_settings.get("min_chars", 25), enabled=config.language_detection),
            "pages": [{"page_number": 1, "blocks": [block], "metadata": {}}],
            "blocks": [block],
            "sections": [],
            "metadata": {
                "source_path": str(loaded_document.path),
                "file_type": loaded_document.extension,
                "mime_type": loaded_document.mime_type,
            },
        }


class ExcelParser(BaseDocumentParser):
    supported_extensions = (".xlsx", ".xls")

    def parse(self, loaded_document: LoadedDocument, config: Optional[ProcessingConfig] = None) -> Dict[str, Any]:
        config = config or ProcessingConfig()
        if pd is None:
            raise ParsingFailureError("pandas is required for Excel parsing.", document=str(loaded_document.path), stage="parsing")

        try:
            workbook = pd.ExcelFile(loaded_document.path)
        except Exception as exc:
            raise ParsingFailureError("Failed to open Excel workbook.", document=str(loaded_document.path), stage="parsing", cause=exc)

        page_blocks: List[Dict[str, Any]] = []
        for sheet_index, sheet_name in enumerate(workbook.sheet_names):
            frame = pd.read_excel(loaded_document.path, sheet_name=sheet_name)
            block = {
                "block_type": "table",
                "text": frame.to_string(index=False),
                "heading_path": [],
                "heading_level": None,
                "page_number": sheet_index + 1,
                "source_location": {"page": sheet_index + 1, "sheet": sheet_name},
                "metadata": {
                    "worksheet": sheet_name,
                    "headers": [safe_text(col) for col in frame.columns],
                    "row_count": len(frame.index),
                    "column_count": len(frame.columns),
                    "extraction_method": "structured_excel",
                },
            }
            page_blocks.append(block)

        all_text = "\n\n".join(block["text"] for block in page_blocks)
        return {
            "document_id": str(uuid4()),
            "title": loaded_document.path.stem,
            "document_type": "excel",
            "language": detect_language(all_text, min_chars=config.language_settings.get("min_chars", 25), enabled=config.language_detection),
            "pages": [
                {"page_number": index + 1, "blocks": [block], "metadata": {"worksheet": block["metadata"]["worksheet"]}}
                for index, block in enumerate(page_blocks)
            ],
            "blocks": page_blocks,
            "sections": [],
            "metadata": {
                "source_path": str(loaded_document.path),
                "file_type": loaded_document.extension,
                "mime_type": loaded_document.mime_type,
            },
        }


class DOCXParser(BaseDocumentParser):
    supported_extensions = (".docx",)

    def parse(self, loaded_document: LoadedDocument, config: Optional[ProcessingConfig] = None) -> Dict[str, Any]:
        config = config or ProcessingConfig()
        if DocxDocument is None:
            raise ParsingFailureError("python-docx is required for DOCX parsing.", document=str(loaded_document.path), stage="parsing")

        doc = DocxDocument(str(loaded_document.path))
        blocks: List[Dict[str, Any]] = []
        heading_stack: List[str] = []

        for paragraph in doc.paragraphs:
            text = (paragraph.text or "").strip()
            if not text:
                continue
            style_name = getattr(paragraph.style, "name", "") if hasattr(paragraph, "style") else ""
            heading_level = None
            if style_name.startswith("Heading"):
                heading_level = 1
                match = re.search(r"(\d+)", style_name)
                if match:
                    heading_level = int(match.group(1))
                if heading_level == 1:
                    heading_stack = [text]
                else:
                    while len(heading_stack) >= heading_level:
                        heading_stack.pop()
                    heading_stack.append(text)
                block_type = "heading"
            elif style_name.lower().startswith("list "):
                block_type = "list"
            else:
                block_type = "paragraph"

            blocks.append(
                {
                    "block_type": block_type,
                    "text": text,
                    "heading_path": list(heading_stack),
                    "heading_level": heading_level,
                    "page_number": 1,
                    "source_location": {"page": 1, "paragraph_index": len(blocks)},
                    "metadata": {"style_name": style_name, "extraction_method": "docx"},
                }
            )

        for table_index, table in enumerate(doc.tables):
            rows = [" | ".join(cell.text for cell in row.cells if cell.text) for row in table.rows]
            table_text = "\n".join(rows)
            blocks.append(
                {
                    "block_type": "table",
                    "text": table_text,
                    "heading_path": list(heading_stack),
                    "heading_level": None,
                    "page_number": 1,
                    "source_location": {"page": 1, "table_index": table_index},
                    "metadata": {"extraction_method": "docx_table"},
                }
            )

        all_text = "\n\n".join(block["text"] for block in blocks)
        return {
            "document_id": str(uuid4()),
            "title": loaded_document.path.stem,
            "document_type": "docx",
            "language": detect_language(all_text, min_chars=config.language_settings.get("min_chars", 25), enabled=config.language_detection),
            "pages": [{"page_number": 1, "blocks": blocks, "metadata": {}}],
            "blocks": blocks,
            "sections": [],
            "metadata": {
                "source_path": str(loaded_document.path),
                "file_type": loaded_document.extension,
                "mime_type": loaded_document.mime_type,
            },
        }


class PDFParser(BaseDocumentParser):
    supported_extensions = (".pdf",)

    def parse(self, loaded_document: LoadedDocument, config: Optional[ProcessingConfig] = None) -> Dict[str, Any]:
        config = config or ProcessingConfig()
        if PdfReader is None:
            raise ParsingFailureError("pypdf is required for PDF parsing.", document=str(loaded_document.path), stage="parsing")

        pages: List[Dict[str, Any]] = []
        flattened_blocks: List[Dict[str, Any]] = []
        all_text_parts: List[str] = []

        try:
            reader = PdfReader(str(loaded_document.path))
            for page_index, page in enumerate(reader.pages, start=1):
                raw_text = ""
                try:
                    raw_text = page.extract_text() or ""
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to extract text from page %s: %s", page_index, exc)
                    raw_text = ""

                blocks = self._convert_text_to_blocks(raw_text, page_index)
                if not blocks and fitz is not None:
                    blocks = self._ocr_page_fallback(loaded_document, page_index, config)
                if not blocks:
                    blocks = [
                        {
                            "block_type": "other",
                            "text": "",
                            "heading_path": [],
                            "heading_level": None,
                            "page_number": page_index,
                            "source_location": {"page": page_index},
                            "metadata": {"extraction_method": "pdf_unreadable"},
                        }
                    ]

                pages.append({"page_number": page_index, "blocks": blocks, "metadata": {"extraction_method": blocks[0].get("metadata", {}).get("extraction_method", "pdf_text")}})
                for block in blocks:
                    flattened_blocks.append(block)
                    if block.get("text"):
                        all_text_parts.append(block["text"])
        except Exception as exc:
            raise ParsingFailureError("Failed to parse PDF document.", document=str(loaded_document.path), stage="parsing", cause=exc)

        return {
            "document_id": str(uuid4()),
            "title": loaded_document.path.stem,
            "document_type": "pdf",
            "language": detect_language("\n\n".join(all_text_parts), min_chars=config.language_settings.get("min_chars", 25), enabled=config.language_detection),
            "pages": pages,
            "blocks": flattened_blocks,
            "sections": [],
            "metadata": {
                "source_path": str(loaded_document.path),
                "file_type": loaded_document.extension,
                "mime_type": loaded_document.mime_type,
            },
        }

    def _convert_text_to_blocks(self, text: str, page_number: int) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        blocks: List[Dict[str, Any]] = []
        for item in re.split(r"\n{2,}", text.strip()):
            cleaned = safe_text(item)
            if not cleaned:
                continue
            heading_path = []
            heading_level = None
            block_type = "paragraph"
            if re.match(r"^(?:#+\s*)?[A-Z0-9][^\n]{1,80}$", cleaned) and len(cleaned.split()) <= 12:
                block_type = "heading"
                heading_path = [cleaned]
                heading_level = 1
            blocks.append(
                {
                    "block_type": block_type,
                    "text": cleaned,
                    "heading_path": heading_path,
                    "heading_level": heading_level,
                    "page_number": page_number,
                    "source_location": {"page": page_number},
                    "metadata": {"extraction_method": "pdf_text"},
                }
            )
        return blocks

    def _ocr_page_fallback(self, loaded_document: LoadedDocument, page_number: int, config: ProcessingConfig) -> List[Dict[str, Any]]:
        if fitz is None or Image is None or pytesseract is None:
            return []
        try:
            doc = fitz.open(str(loaded_document.path))
            if page_number - 1 >= len(doc):
                return []
            page = doc[page_number - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            mode = "RGB" if pix.n < 4 else "RGBA"
            image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            ocr_languages = config.ocr_languages or config.ocr_settings.get("languages", ["eng"])
            text = pytesseract.image_to_string(image, lang=",".join(ocr_languages))
            if not text.strip():
                return []
            return [
                {
                    "block_type": "paragraph",
                    "text": text.strip(),
                    "heading_path": [],
                    "heading_level": None,
                    "page_number": page_number,
                    "source_location": {"page": page_number},
                    "metadata": {"extraction_method": "ocr", "ocr_languages": ocr_languages, "ocr_confidence": None},
                }
            ]
        except Exception as exc:  # pragma: no cover
            logger.warning("OCR fallback failed for PDF page %s: %s", page_number, exc)
            return []


class ImageParser(BaseDocumentParser):
    supported_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".tif")

    def parse(self, loaded_document: LoadedDocument, config: Optional[ProcessingConfig] = None) -> Dict[str, Any]:
        config = config or ProcessingConfig()
        if Image is None or pytesseract is None:
            raise OCRFailureError("Pillow and pytesseract are required for OCR image parsing.", document=str(loaded_document.path), stage="ocr")

        try:
            image = Image.open(loaded_document.path)
        except Exception as exc:
            raise OCRFailureError("Failed to open image file for OCR.", document=str(loaded_document.path), stage="ocr", cause=exc)

        ocr_languages = config.ocr_languages or config.ocr_settings.get("languages", ["eng"])
        text = pytesseract.image_to_string(image, lang=",".join(ocr_languages)) or ""
        block = {
            "block_type": "paragraph",
            "text": text.strip(),
            "heading_path": [],
            "heading_level": None,
            "page_number": 1,
            "source_location": {"page": 1},
            "metadata": {"extraction_method": "ocr", "ocr_languages": ocr_languages, "ocr_confidence": None},
        }
        return {
            "document_id": str(uuid4()),
            "title": loaded_document.path.stem,
            "document_type": "image",
            "language": detect_language(text, min_chars=config.language_settings.get("min_chars", 25), enabled=config.language_detection),
            "pages": [{"page_number": 1, "blocks": [block], "metadata": {"extraction_method": "ocr"}}],
            "blocks": [block],
            "sections": [],
            "metadata": {
                "source_path": str(loaded_document.path),
                "file_type": loaded_document.extension,
                "mime_type": loaded_document.mime_type,
            },
        }


class ParserRouter:
    """Selects the correct parser based on file type."""

    def __init__(self, parsers: Optional[Sequence[BaseDocumentParser]] = None) -> None:
        self.parsers = parsers or [
            PDFParser(),
            DOCXParser(),
            CSVParser(),
            ExcelParser(),
            ImageParser(),
            PlainTextParser(),
        ]

    def select_parser(self, loaded_document: LoadedDocument) -> BaseDocumentParser:
        for parser in self.parsers:
            if loaded_document.extension.lower() in parser.supported_extensions:
                return parser
        raise UnsupportedFormatError(
            f"Unsupported document format '{loaded_document.extension}'.",
            document=str(loaded_document.path),
            stage="parser_selection",
        )

    def parse(self, loaded_document: LoadedDocument, config: Optional[ProcessingConfig] = None) -> Dict[str, Any]:
        parser = self.select_parser(loaded_document)
        return build_document_hierarchy(parser.parse(loaded_document, config))


class FixedSizeChunkingStrategy:
    """Fallback chunking strategy with overlap and tail handling."""

    def chunk(self, document: Dict[str, Any], config: Optional[ProcessingConfig] = None) -> List[Chunk]:
        config = config or ProcessingConfig()
        blocks = document.get("blocks") or []
        chunks: List[Chunk] = []

        for block in blocks:
            text = safe_text(block.get("text", ""))
            if not text:
                continue
            heading_path = block.get("heading_path") or []
            content = self._render_with_heading(heading_path, text)
            step = max(1, config.chunk_size - config.overlap)
            for start in range(0, len(content), step):
                piece = content[start : start + config.chunk_size]
                if not piece.strip():
                    continue
                if len(piece.strip()) < config.min_chunk_size and chunks:
                    if len(chunks[-1].content) + len(piece) <= config.max_chunk_size:
                        chunks[-1].content = (chunks[-1].content.rstrip() + "\n\n" + piece.strip()).strip()
                        chunks[-1].metadata["character_count"] = len(chunks[-1].content)
                        chunks[-1].metadata["token_count"] = estimate_token_count(chunks[-1].content)
                        continue
                chunk = Chunk(
                    content=piece.strip(),
                    metadata={
                        "chunk_index": len(chunks),
                        "document_id": document.get("document_id"),
                        "content_type": block.get("block_type", "paragraph"),
                        "source_location": block.get("source_location", {}),
                        "heading_path": list(heading_path),
                        "page_number": block.get("page_number"),
                        "section": None,
                        "language": document.get("language", "unknown"),
                        "extraction_method": block.get("metadata", {}).get("extraction_method", "text"),
                        "chunking_strategy": "fixed_size",
                        "character_count": len(piece.strip()),
                        "token_count": estimate_token_count(piece.strip()),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                chunks.append(chunk)

        return chunks

    def _render_with_heading(self, heading_path: Sequence[str], text: str) -> str:
        if heading_path:
            return "\n\n".join([*heading_path, text])
        return text


class StructureAwareChunkingStrategy:
    """Preserves heading hierarchy and logical section context."""

    def chunk(self, document: Dict[str, Any], config: Optional[ProcessingConfig] = None) -> List[Chunk]:
        config = config or ProcessingConfig()
        blocks = self._collect_blocks(document)
        chunks: List[Chunk] = []
        current_heading: List[str] = []
        current_parts: List[str] = []

        def flush_current() -> None:
            if not current_parts:
                return
            payload = self._assemble_payload(current_heading, current_parts)
            if not payload:
                return
            chunks.append(
                Chunk(
                    content=payload.strip(),
                    metadata={
                        "chunk_index": len(chunks),
                        "document_id": document.get("document_id"),
                        "content_type": "structured",
                        "heading_path": list(current_heading),
                        "section": None,
                        "language": document.get("language", "unknown"),
                        "chunking_strategy": "structure_aware",
                        "character_count": len(payload.strip()),
                        "token_count": estimate_token_count(payload.strip()),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            )

        for block in blocks:
            text = safe_text(block.get("text", ""))
            if not text:
                continue
            block_heading = block.get("heading_path") or []
            if block.get("block_type") == "heading":
                if current_parts:
                    flush_current()
                current_heading = list(block_heading) if block_heading else [text]
                current_parts = [text]
                continue

            if block_heading:
                current_heading = list(block_heading)
            if current_parts:
                current_parts.append(text)
            else:
                current_parts = [text]

            combined = self._assemble_payload(current_heading, current_parts)
            if len(combined) > config.chunk_size:
                flush_current()
                current_parts = [text]

        if current_parts:
            flush_current()

        return chunks if chunks else FixedSizeChunkingStrategy().chunk(document, config)

    def _collect_blocks(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocks = document.get("blocks") or []
        if blocks:
            return blocks

        collected: List[Dict[str, Any]] = []
        for page in document.get("pages") or []:
            page_blocks = page.get("blocks") or []
            for block in page_blocks:
                if block:
                    collected.append(block)
        return collected

    def _assemble_payload(self, heading_path: Sequence[str], parts: Sequence[str]) -> str:
        combined = "\n\n".join(part.strip() for part in parts if part and part.strip())
        if heading_path:
            return "\n\n".join([*heading_path, combined])
        return combined


class SentenceAwareChunkingStrategy:
    """Chunk text at sentence boundaries wherever possible."""

    def chunk(self, document: Dict[str, Any], config: Optional[ProcessingConfig] = None) -> List[Chunk]:
        config = config or ProcessingConfig()
        chunks: List[Chunk] = []
        for block in document.get("blocks") or []:
            text = safe_text(block.get("text", ""))
            if not text:
                continue
            heading_path = block.get("heading_path") or []
            segments = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", text) if seg.strip()]
            if not segments:
                continue
            current: List[str] = []
            for segment in segments:
                candidate = " ".join([*current, segment])
                if len(candidate) > config.chunk_size and current:
                    chunks.append(self._build_chunk(current, heading_path, document, chunks, "sentence_aware"))
                    current = [segment]
                else:
                    current.append(segment)
            if current:
                chunks.append(self._build_chunk(current, heading_path, document, chunks, "sentence_aware"))
        return chunks if chunks else FixedSizeChunkingStrategy().chunk(document, config)

    def _build_chunk(self, sentences: Sequence[str], heading_path: Sequence[str], document: Dict[str, Any], existing: List[Chunk], strategy_name: str) -> Chunk:
        content = " ".join(sentences).strip()
        if heading_path:
            content = "\n\n".join([*heading_path, content])
        return Chunk(
            content=content,
            metadata={
                "chunk_index": len(existing),
                "document_id": document.get("document_id"),
                "content_type": "sentence",
                "heading_path": list(heading_path),
                "section": None,
                "language": document.get("language", "unknown"),
                "chunking_strategy": strategy_name,
                "character_count": len(content),
                "token_count": estimate_token_count(content),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )


class ParagraphAwareChunkingStrategy:
    """Chunk text at paragraph boundaries before falling back to sentence logic."""

    def chunk(self, document: Dict[str, Any], config: Optional[ProcessingConfig] = None) -> List[Chunk]:
        config = config or ProcessingConfig()
        chunks: List[Chunk] = []
        for block in document.get("blocks") or []:
            text = safe_text(block.get("text", ""))
            if not text:
                continue
            paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
            if not paragraphs:
                continue
            heading_path = block.get("heading_path") or []
            current: List[str] = []
            for paragraph in paragraphs:
                candidate = "\n\n".join([*current, paragraph]) if current else paragraph
                if len(candidate) > config.chunk_size and current:
                    chunks.append(self._build_chunk(current, heading_path, document, chunks, "paragraph_aware"))
                    current = [paragraph]
                else:
                    current.append(paragraph)
            if current:
                chunks.append(self._build_chunk(current, heading_path, document, chunks, "paragraph_aware"))
        return chunks if chunks else FixedSizeChunkingStrategy().chunk(document, config)

    def _build_chunk(self, paragraphs: Sequence[str], heading_path: Sequence[str], document: Dict[str, Any], existing: List[Chunk], strategy_name: str) -> Chunk:
        content = "\n\n".join(paragraphs).strip()
        if heading_path:
            content = "\n\n".join([*heading_path, content])
        return Chunk(
            content=content,
            metadata={
                "chunk_index": len(existing),
                "document_id": document.get("document_id"),
                "content_type": "paragraph",
                "heading_path": list(heading_path),
                "section": None,
                "language": document.get("language", "unknown"),
                "chunking_strategy": strategy_name,
                "character_count": len(content),
                "token_count": estimate_token_count(content),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )


class TableAwareChunkingStrategy:
    """Chunk tables by row groups while keeping headers and title context."""

    def chunk(self, document: Dict[str, Any], config: Optional[ProcessingConfig] = None) -> List[Chunk]:
        config = config or ProcessingConfig()
        chunks: List[Chunk] = []
        row_group_size = max(1, int(config.table_row_group_size or 25))
        for block in document.get("blocks") or []:
            table_text = safe_text(block.get("text", ""))
            if not table_text:
                continue
            if block.get("block_type") != "table" and not block.get("metadata", {}).get("headers"):
                continue
            heading_path = block.get("heading_path") or []
            rows = [line.strip() for line in table_text.splitlines() if line.strip()]
            if not rows:
                continue
            header = rows[0]
            data_rows = rows[1:]
            for idx in range(0, len(data_rows), row_group_size):
                group = data_rows[idx : idx + row_group_size]
                payload = "\n".join([*([header] if header else []), *group])
                if heading_path:
                    payload = "\n\n".join([*heading_path, payload])
                chunks.append(
                    Chunk(
                        content=payload,
                        metadata={
                            "chunk_index": len(chunks),
                            "document_id": document.get("document_id"),
                            "content_type": "table",
                            "heading_path": list(heading_path),
                            "section": None,
                            "language": document.get("language", "unknown"),
                            "chunking_strategy": "table_aware",
                            "character_count": len(payload),
                            "token_count": estimate_token_count(payload),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "source_location": block.get("source_location", {}),
                        },
                    )
                )
        return chunks if chunks else FixedSizeChunkingStrategy().chunk(document, config)


class ParentChildChunkingStrategy:
    """Create a light parent-child relationship around section chunks."""

    def chunk(self, document: Dict[str, Any], config: Optional[ProcessingConfig] = None) -> List[Chunk]:
        config = config or ProcessingConfig()
        base_chunks = StructureAwareChunkingStrategy().chunk(document, config)
        if not base_chunks:
            base_chunks = FixedSizeChunkingStrategy().chunk(document, config)

        parent_chunks: List[Chunk] = []
        child_chunks: List[Chunk] = []
        for idx, chunk in enumerate(base_chunks):
            parent_id = f"{document.get('document_id', 'doc')}-parent-{idx}"
            parent_payload = chunk.content.splitlines()[0] if chunk.content else "Parent context"
            parent_meta = {
                **chunk.metadata,
                "chunk_index": len(parent_chunks),
                "parent_chunk_id": None,
                "chunking_strategy": "parent_child",
                "content_type": "section",
            }
            parent_chunk = Chunk(content=parent_payload, metadata=parent_meta)
            parent_chunks.append(parent_chunk)

            child_meta = {
                **chunk.metadata,
                "chunk_index": len(child_chunks),
                "parent_chunk_id": parent_id,
                "chunking_strategy": "parent_child_child",
                "content_type": "section_child",
            }
            child_chunks.append(Chunk(content=chunk.content, metadata=child_meta))

        return parent_chunks + child_chunks


class RecursiveChunkingStrategy:
    """Recursive fallback that uses document structure before falling back."""

    def chunk(self, document: Dict[str, Any], config: Optional[ProcessingConfig] = None) -> List[Chunk]:
        config = config or ProcessingConfig()
        strategy = StructureAwareChunkingStrategy()
        chunks = strategy.chunk(document, config)
        return chunks if chunks else FixedSizeChunkingStrategy().chunk(document, config)


class ChunkingEngine:
    """Dispatcher that selects a chunking strategy."""

    def __init__(self, config: Optional[ProcessingConfig] = None) -> None:
        self.config = config or ProcessingConfig()

    def _choose_automatic_strategy(self, document: Optional[Dict[str, Any]], config: ProcessingConfig):
        profile = (config.chunking_profile or "balanced").lower()
        if profile == "table_focused":
            return TableAwareChunkingStrategy()
        if profile == "sentence_focused":
            return SentenceAwareChunkingStrategy()
        if profile == "paragraph_focused":
            return ParagraphAwareChunkingStrategy()
        if profile == "structured":
            return StructureAwareChunkingStrategy()

        if document is not None:
            blocks = document.get("blocks") or []
            if not blocks:
                for page in document.get("pages") or []:
                    blocks.extend(page.get("blocks") or [])

            has_table = any(
                (block.get("block_type") == "table" or block.get("metadata", {}).get("headers"))
                for block in blocks
            )
            has_heading = any(block.get("block_type") == "heading" for block in blocks)
            if has_table and not has_heading:
                return TableAwareChunkingStrategy()
            if has_heading or profile == "structured":
                return StructureAwareChunkingStrategy()

        return StructureAwareChunkingStrategy()

    def select_strategy(self, strategy_name: Optional[str] = None, *, config: Optional[ProcessingConfig] = None, document: Optional[Dict[str, Any]] = None):
        effective_config = config or self.config
        if effective_config.parent_child_mode:
            return ParentChildChunkingStrategy()

        strategy_name = (strategy_name or effective_config.chunking_strategy or "balanced").lower()
        if strategy_name == "auto":
            return self._choose_automatic_strategy(document, effective_config)

        mapping = {
            "fixed": FixedSizeChunkingStrategy,
            "fixed_size": FixedSizeChunkingStrategy,
            "recursive": RecursiveChunkingStrategy,
            "structure": StructureAwareChunkingStrategy,
            "structure_aware": StructureAwareChunkingStrategy,
            "sentence": SentenceAwareChunkingStrategy,
            "sentence_aware": SentenceAwareChunkingStrategy,
            "paragraph": ParagraphAwareChunkingStrategy,
            "paragraph_aware": ParagraphAwareChunkingStrategy,
            "table": TableAwareChunkingStrategy,
            "table_aware": TableAwareChunkingStrategy,
            "parent_child": ParentChildChunkingStrategy,
            "balanced": StructureAwareChunkingStrategy,
        }
        return mapping.get(strategy_name, StructureAwareChunkingStrategy)()

    def chunk(self, document: Dict[str, Any], config: Optional[ProcessingConfig] = None) -> List[Chunk]:
        config = config or self.config
        strategy = self.select_strategy(config.chunking_strategy, config=config, document=document)
        return strategy.chunk(document, config)


class DocumentProcessor:
    """High-level document pipeline: load → parse → normalize → chunk."""

    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        router: Optional[ParserRouter] = None,
        chunking_engine: Optional[ChunkingEngine] = None,
        finalization_layer: Optional[ChunkFinalizationLayer] = None,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.router = router or ParserRouter()
        self.chunking_engine = chunking_engine or ChunkingEngine()
        self.finalization_layer = finalization_layer or ChunkFinalizationLayer()

    def process(self, file_path: Union[str, Path], *, config: Optional[ProcessingConfig] = None) -> Tuple[Dict[str, Any], List[FinalizedChunk]]:
        config = config or ProcessingConfig()
        loaded_document = self.loader.load(file_path)
        normalized_document = self.router.parse(loaded_document, config)
        raw_chunks = self.chunking_engine.chunk(normalized_document, config)
        chunks = self.finalization_layer.finalize(raw_chunks, document=normalized_document, config=config)
        return normalized_document, chunks


@dataclass
class DocumentAsset:
    """Backward-compatible document metadata container."""

    doc_id: str
    filename: str
    category: str
    chunk_count: int
    ingested_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    _supabase_synced: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "category": self.category,
            "chunk_count": self.chunk_count,
            "ingested_at": self.ingested_at,
            "metadata": self.metadata,
            "_supabase_synced": self._supabase_synced,
        }


class DocumentManager:
    """Backward-compatible API that delegates to the modular pipeline."""

    SUPPORTED_FORMATS = DEFAULT_SUPPORTED_EXTENSIONS
    CHUNK_SIZE = 1600
    CHUNK_OVERLAP = 200
    MIN_CHUNK_SIZE = 50

    def __init__(
        self,
        rag: Optional[Any] = None,
        supabase_client: Optional[Any] = None,
        *,
        loader: Optional[DocumentLoader] = None,
        router: Optional[ParserRouter] = None,
        chunking_engine: Optional[ChunkingEngine] = None,
        finalization_layer: Optional[ChunkFinalizationLayer] = None,
    ) -> None:
        self.rag = rag
        self.supabase_client = supabase_client
        self.documents: Dict[str, DocumentAsset] = {}
        self.processor = DocumentProcessor(
            loader=loader,
            router=router,
            chunking_engine=chunking_engine,
            finalization_layer=finalization_layer,
        )
        self.supported_formats = self.SUPPORTED_FORMATS.copy()

    def ingest(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        *,
        processing_config: Optional[ProcessingConfig] = None,
    ) -> DocumentAsset:
        metadata = metadata or {}
        processing_config = processing_config or ProcessingConfig(
            chunk_size=self.CHUNK_SIZE,
            overlap=self.CHUNK_OVERLAP,
            min_chunk_size=self.MIN_CHUNK_SIZE,
        )

        loaded_document = self.processor.loader.load(file_path)
        normalized_document = self.processor.router.parse(loaded_document, processing_config)
        raw_chunks = self.processor.chunking_engine.chunk(normalized_document, processing_config)
        chunks = self.processor.finalization_layer.finalize(raw_chunks, document=normalized_document, config=processing_config)
        if not chunks:
            raise ChunkingFailureError("The document did not produce any valid chunks.", document=str(loaded_document.path), stage="chunking")

        doc_id = str(uuid4())
        ingested_at = datetime.now(timezone.utc).isoformat()
        category = metadata.get("category", "General")

        rag_documents: List[Tuple[str, Dict[str, Any]]] = []
        for chunk_index, chunk in enumerate(chunks):
            chunk_meta = {
                "doc_id": doc_id,
                "filename": loaded_document.path.name,
                "source_path": str(loaded_document.path),
                "chunk_index": chunk_index,
                "total_chunks": len(chunks),
                "category": category,
                "priority": metadata.get("priority", "Medium"),
                "due_date": metadata.get("due_date"),
                "user_goal": metadata.get("user_goal", ""),
                "language": chunk.metadata.get("language", normalized_document.get("language", "unknown")),
                "ingested_at": ingested_at,
                "document_type": normalized_document.get("document_type", loaded_document.extension),
                "content_type": chunk.metadata.get("content_type", "paragraph"),
                "page_number": chunk.metadata.get("page_number"),
                "section": chunk.metadata.get("section"),
                "heading_path": chunk.metadata.get("heading_path", []),
                "extraction_method": chunk.metadata.get("extraction_method", "text"),
                "chunking_strategy": chunk.metadata.get("chunking_strategy", processing_config.chunking_strategy),
                "character_count": chunk.metadata.get("character_count", len(chunk.content)),
                "token_count": chunk.metadata.get("token_count", estimate_token_count(chunk.content)),
                "source_location": chunk.metadata.get("source_location", {}),
                **{k: v for k, v in metadata.items() if k not in {"category", "priority", "due_date", "user_goal"}},
            }
            rag_documents.append((chunk.embedding_text, chunk_meta))

        if self.rag is not None and hasattr(self.rag, "add_documents"):
            try:
                self.rag.add_documents(rag_documents)
            except Exception as exc:  # pragma: no cover
                logger.warning("Local indexing failed for %s: %s", loaded_document.path.name, exc)

        asset = DocumentAsset(
            doc_id=doc_id,
            filename=loaded_document.path.name,
            category=category,
            chunk_count=len(chunks),
            ingested_at=ingested_at,
            metadata={**metadata, "document_type": normalized_document.get("document_type"), "language": normalized_document.get("language", "unknown")},
            _supabase_synced=True,
        )
        self.documents[doc_id] = asset
        return asset

    def get_document(self, doc_id: str) -> Optional[DocumentAsset]:
        return self.documents.get(doc_id)

    def list_documents(self, category: Optional[str] = None) -> List[DocumentAsset]:
        docs = list(self.documents.values())
        if category:
            return [doc for doc in docs if doc.category == category]
        return docs

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        return True


__all__ = [
    "DocumentLoader",
    "DocumentManager",
    "DocumentAsset",
    "ProcessingConfig",
    "Chunk",
    "FinalizedChunk",
    "ChunkFinalizationLayer",
    "FixedSizeChunkingStrategy",
    "StructureAwareChunkingStrategy",
    "SentenceAwareChunkingStrategy",
    "ParagraphAwareChunkingStrategy",
    "TableAwareChunkingStrategy",
    "ParentChildChunkingStrategy",
    "RecursiveChunkingStrategy",
    "ChunkingEngine",
    "ParserRouter",
    "BaseDocumentParser",
    "PDFParser",
    "DOCXParser",
    "CSVParser",
    "ExcelParser",
    "ImageParser",
    "PlainTextParser",
    "UnsupportedFormatError",
    "InvalidFileError",
    "FileTooLargeError",
    "ParsingFailureError",
    "OCRFailureError",
    "NormalizationFailureError",
    "ChunkingFailureError",
    "InvalidChunkError",
]
