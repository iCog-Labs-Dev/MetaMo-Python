from typing import List

from core.config import PAPER_CHUNK_SIZE, PAPER_CHUNK_OVERLAP, PAPER_MAX_CHARS
from core.state import DocumentChunk
from core.interfaces import ChunkingStrategy


class OverlapChunker(ChunkingStrategy):
    def __init__(self, chunk_size: int = PAPER_CHUNK_SIZE, overlap: int = PAPER_CHUNK_OVERLAP):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source_path: str) -> List[DocumentChunk]:
        text = text[:PAPER_MAX_CHARS]
        chunks: list[DocumentChunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(
                DocumentChunk(
                    index=index,
                    text=text[start:end],
                    source_path=source_path,
                    char_start=start,
                    char_end=end,
                )
            )
            index += 1
            if end == len(text):
                break
            start += self.chunk_size - self.overlap
        return chunks
