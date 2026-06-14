import json
import os
from typing import List, Optional

from core.config import PAPER_STORAGE_PATH
from core.state import DocumentChunk, Paper
from core.interfaces import PaperRepository


def _paper_path(store_dir: str, paper_id: str) -> str:
    return os.path.join(store_dir, f"{paper_id}.json")


def _paper_to_dict(p: Paper) -> dict:
    return {
        "paper_id": p.paper_id,
        "source_path": p.source_path,
        "title": p.title,
        "total_chars": p.total_chars,
        "ingested_at": p.ingested_at,
        "chunks": [
            {
                "index": c.index,
                "text": c.text,
                "source_path": c.source_path,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for c in p.chunks
        ],
    }


def _dict_to_paper(d: dict) -> Paper:
    p = Paper(
        source_path=d["source_path"],
        title=d["title"],
        total_chars=d["total_chars"],
        ingested_at=d.get("ingested_at"),
        paper_id=d["paper_id"],
    )
    p.chunks = [
        DocumentChunk(
            index=c["index"],
            text=c["text"],
            source_path=c["source_path"],
            char_start=c["char_start"],
            char_end=c["char_end"],
        )
        for c in d.get("chunks", [])
    ]
    return p


class JsonPaperRepository(PaperRepository):
    def __init__(self, store_dir: str = PAPER_STORAGE_PATH):
        self._dir = store_dir
        os.makedirs(self._dir, exist_ok=True)

    def _paper_path(self, paper_id: str) -> str:
        return _paper_path(self._dir, paper_id)

    def _all_paper_ids(self) -> List[str]:
        if not os.path.isdir(self._dir):
            return []
        return sorted(
            f.removesuffix(".json")
            for f in os.listdir(self._dir)
            if f.endswith(".json")
        )

    def save(self, paper: Paper) -> str:
        path = self._paper_path(paper.paper_id)
        with open(path, "w") as f:
            json.dump(_paper_to_dict(paper), f, indent=2)
        return paper.paper_id

    def load(self, paper_id: str) -> Optional[Paper]:
        path = self._paper_path(paper_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return None
            return _dict_to_paper(data)
        except (json.JSONDecodeError, KeyError, IOError):
            return None

    def list_papers(self) -> List[Paper]:
        papers = []
        for pid in self._all_paper_ids():
            p = self.load(pid)
            if p is not None:
                papers.append(p)
        return papers

    def delete(self, paper_id: str) -> bool:
        path = self._paper_path(paper_id)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True
