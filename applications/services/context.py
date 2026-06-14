from typing import List, Optional

from core.state import Paper
from core.interfaces import PaperRepository
from applications.services.storage import JsonPaperRepository


class PaperContextService:
    def __init__(self, repository: Optional[PaperRepository] = None):
        self.repository = repository or JsonPaperRepository()

    def list_papers(self) -> List[Paper]:
        return self.repository.list_papers()

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        return self.repository.load(paper_id)

    def get_latest_paper(self) -> Optional[Paper]:
        papers = self.repository.list_papers()
        if not papers:
            return None
        papers.sort(key=lambda p: p.ingested_at or "", reverse=True)
        return papers[0]

    def get_recent_papers(self, limit: int = 3) -> List[Paper]:
        papers = self.repository.list_papers()
        papers.sort(key=lambda p: p.ingested_at or "", reverse=True)
        return papers[:limit]

    def has_papers(self) -> bool:
        return len(self.repository.list_papers()) > 0
