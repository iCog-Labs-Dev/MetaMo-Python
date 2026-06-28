import os
from datetime import datetime
from typing import Optional

from applications.papers.config import PAPER_CONTEXT_MAX_CHARS
from applications.papers.entities import Paper
from applications.papers.interfaces import PaperRepository, TextExtractor, ChunkingStrategy
from applications.papers.services.extractors import get_extractor
from applications.papers.services.chunker import OverlapChunker
from applications.papers.services.storage import JsonPaperRepository


class PaperIngestionService:
    """Orchestrates paper ingestion: extraction, chunking, and persistence."""

    def __init__(
        self,
        extractor: Optional[TextExtractor] = None,
        chunker: Optional[ChunkingStrategy] = None,
        repository: Optional[PaperRepository] = None,
    ):
        self.extractor = extractor
        self.chunker = chunker or OverlapChunker()
        self.repository = repository or JsonPaperRepository()

    def _get_extractor(self, path: str) -> TextExtractor:
        """Return the configured extractor, or auto-detect one for *path*."""
        return self.extractor or get_extractor(path)

    def ingest(self, path: str) -> Paper:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        extractor = self._get_extractor(path)
        text = extractor.extract(path)
        chunks = self.chunker.chunk(text, path)

        paper = Paper(
            source_path=path,
            title=os.path.splitext(os.path.basename(path))[0],
            total_chars=len(text),
            chunks=chunks,
            ingested_at=datetime.now().isoformat(),
        )

        self.repository.save(paper)
        return paper

    def build_context(self, paper: Paper, max_chars: int = PAPER_CONTEXT_MAX_CHARS) -> str:
        """Concatenate paper chunks up to *max_chars*, appending a truncation notice if needed."""
        accumulated = []
        total = 0
        for chunk in sorted(paper.chunks, key=lambda c: c.index):
            if total + len(chunk.text) > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    accumulated.append(chunk.text[:remaining])
                break
            accumulated.append(chunk.text)
            total += len(chunk.text)

        full = "".join(accumulated)
        if len(full) < paper.total_chars:
            full += "\n... [content truncated]"
        return full
