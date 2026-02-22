"""PDF text extraction from ClearPath documentation files."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single page of extracted text from a PDF."""
    filename: str
    page_number: int
    text: str
    metadata: dict = field(default_factory=dict)


def extract_text_from_pdf(pdf_path: Path) -> List[Document]:
    """Extract text from a single PDF file, page by page."""
    documents = []
    try:
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                # Clean up common extraction artifacts
                text = _clean_text(text)
                documents.append(Document(
                    filename=pdf_path.name,
                    page_number=page_num,
                    text=text,
                    metadata={
                        "total_pages": len(reader.pages),
                        "source_path": str(pdf_path),
                    }
                ))
        if not documents:
            logger.warning(f"No text extracted from {pdf_path.name}")
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path.name}: {e}")
    return documents


def load_all_documents(docs_dir: Path) -> List[Document]:
    """Load all PDF files from the docs directory."""
    all_docs = []
    pdf_files = sorted(docs_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_dir}")
        return all_docs

    logger.info(f"Loading {len(pdf_files)} PDF files from {docs_dir}")

    for pdf_path in pdf_files:
        docs = extract_text_from_pdf(pdf_path)
        all_docs.extend(docs)
        logger.info(f"  {pdf_path.name}: {len(docs)} pages extracted")

    logger.info(f"Total documents loaded: {len(all_docs)} pages from {len(pdf_files)} PDFs")
    return all_docs


def _clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    # Normalize whitespace but preserve paragraph breaks
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
        elif cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")  # Preserve paragraph breaks

    text = "\n".join(cleaned_lines)

    # Remove excessive whitespace within lines
    import re
    text = re.sub(r" {3,}", "  ", text)

    return text.strip()
