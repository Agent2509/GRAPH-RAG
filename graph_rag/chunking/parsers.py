"""Document parsers for extracting raw text from various file formats."""

import io
from pathlib import Path
from typing import Union
from pypdf import PdfReader


def parse_file(file_path: Union[str, Path]) -> str:
    """Read and extract text from a file (.txt, .md, .pdf)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix in [".txt", ".md", ".markdown", ".csv", ".json"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        reader = PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    else:
        # Fallback: try reading as plain text
        return path.read_text(encoding="utf-8", errors="ignore")


def parse_bytes(content: bytes, filename: str) -> str:
    """Extract text from in-memory bytes based on filename extension."""
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        stream = io.BytesIO(content)
        reader = PdfReader(stream)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    else:
        # Assume utf-8 text / markdown / csv
        return content.decode("utf-8", errors="ignore")
