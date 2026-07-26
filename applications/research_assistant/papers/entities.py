from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class DocumentChunk:
    """A contiguous segment of text extracted from a source document."""

    index: int
    text: str
    source_path: str
    char_start: int
    char_end: int


@dataclass
class Paper:
    """Represents an ingested document with its metadata and content chunks."""

    source_path: str
    title: str
    total_chars: int
    chunks: List[DocumentChunk] = field(default_factory=list)
    ingested_at: Optional[str] = None
    paper_id: str = ""

    def __post_init__(self):
        """Auto-generate a short paper ID from source path, char count, and timestamp."""
        if not self.paper_id:
            import hashlib
            raw = f"{self.source_path}:{self.total_chars}:{self.ingested_at or datetime.now().isoformat()}"
            self.paper_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    @property
    def full_text(self) -> str:
        """Return the concatenated text of all chunks in order."""
        return "".join(c.text for c in sorted(self.chunks, key=lambda x: x.index))
