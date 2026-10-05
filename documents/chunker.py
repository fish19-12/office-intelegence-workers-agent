from typing import List


class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                split_point = self._find_split_point(text, start, end)
                end = split_point
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, end - 1)
        return chunks

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        for sep in self.separators:
            if sep == "":
                continue
            slice_text = text[start:end]
            idx = slice_text.rfind(sep)
            if idx != -1:
                return start + idx + len(sep)
        return end


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)
