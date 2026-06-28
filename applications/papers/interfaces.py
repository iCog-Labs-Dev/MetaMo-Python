from abc import ABC, abstractmethod
from typing import List, Optional

from applications.papers.entities import DocumentChunk, Paper


class TextExtractor(ABC):
    """Interface for extracting plain text from a file at a given path."""

    @abstractmethod
    def extract(self, path: str) -> str:
        ...


class ChunkingStrategy(ABC):
    """Interface for splitting text into smaller document chunks."""

    @abstractmethod
    def chunk(self, text: str, source_path: str) -> List[DocumentChunk]:
        ...


class PaperRepository(ABC):
    """Interface for persisting and retrieving Paper objects."""

    @abstractmethod
    def save(self, paper: Paper) -> str:
        """Persist a paper and return its ID."""
        ...

    @abstractmethod
    def load(self, paper_id: str) -> Optional[Paper]:
        """Retrieve a paper by ID, or None if not found."""
        ...

    @abstractmethod
    def list_papers(self) -> List[Paper]:
        """Return all stored papers."""
        ...

    @abstractmethod
    def delete(self, paper_id: str) -> bool:
        """Remove a paper by ID. Returns True if it existed."""
        ...
