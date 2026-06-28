from typing import List

from applications.papers.config import PAPER_CHUNK_SIZE, PAPER_CHUNK_OVERLAP, PAPER_MAX_CHARS
from applications.papers.entities import DocumentChunk
from applications.papers.interfaces import ChunkingStrategy


class OverlapChunker(ChunkingStrategy):
    """Splits text into overlapping chunks of a fixed size."""

    def __init__(self, chunk_size: int = PAPER_CHUNK_SIZE, overlap: int = PAPER_CHUNK_OVERLAP):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source_path: str) -> List[DocumentChunk]:
        """Chunk text into overlapping segments, each linked to the source path."""
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
