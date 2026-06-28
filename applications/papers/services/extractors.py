import subprocess
from applications.papers.interfaces import TextExtractor


class PdfTextExtractor(TextExtractor):
    """Extract text from PDF files using pdftotext."""

    def extract(self, path: str) -> str:
        result = subprocess.run(
            ["pdftotext", path, "-"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftotext failed: {result.stderr}")
        return result.stdout


class TxtTextExtractor(TextExtractor):
    """Extract text from plain-text files by reading the file directly."""

    def extract(self, path: str) -> str:
        with open(path) as f:
            return f.read()


_EXTENSIONS = {
    ".pdf": PdfTextExtractor,
    ".txt": TxtTextExtractor,
}


def get_extractor(path: str) -> TextExtractor:
    """Return the appropriate TextExtractor for the file extension of *path*."""
    import os
    ext = os.path.splitext(path)[1].lower()
    cls = _EXTENSIONS.get(ext)
    if cls is None:
        raise ValueError(f"Unsupported file extension: {ext} (supported: {list(_EXTENSIONS)})")
    return cls()
