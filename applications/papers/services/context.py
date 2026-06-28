from typing import List, Optional

from applications.papers.entities import Paper
from applications.papers.interfaces import PaperRepository
from applications.papers.services.storage import JsonPaperRepository


class PaperContextService:
    """Provides query access to stored papers via a repository."""

    def __init__(self, repository: Optional[PaperRepository] = None):
        self.repository = repository or JsonPaperRepository()

    def list_papers(self) -> List[Paper]:
        """Return all stored papers."""
        return self.repository.list_papers()

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Retrieve a single paper by its ID, or None if not found."""
        return self.repository.load(paper_id)

    def get_latest_paper(self) -> Optional[Paper]:
        """Return the most recently ingested paper, or None if none exist."""
        papers = self.repository.list_papers()
        if not papers:
            return None
        papers.sort(key=lambda p: p.ingested_at or "", reverse=True)
        return papers[0]

    def get_recent_papers(self, limit: int = 3) -> List[Paper]:
        """Return the *limit* most recently ingested papers."""
        papers = self.repository.list_papers()
        papers.sort(key=lambda p: p.ingested_at or "", reverse=True)
        return papers[:limit]

    def has_papers(self) -> bool:
        """Check whether at least one paper exists in the repository."""
        return len(self.repository.list_papers()) > 0
