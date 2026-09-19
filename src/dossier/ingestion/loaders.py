from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class Page:
    number: int  # 1-based
    text: str
    ocr: bool = False


def _ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_pdf_page(page) -> str:
    """OCR a PyMuPDF page in French + Arabic. Requires tesseract with fra/ara packs."""
    import io

    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang="fra+ara")


def load_pdf(path: Path, ocr: bool = True) -> list[Page]:
    import pymupdf

    pages: list[Page] = []
    use_ocr = ocr and _ocr_available()
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            did_ocr = False
            if len(text) < 20 and use_ocr:
                try:
                    text = _ocr_pdf_page(page).strip()
                    did_ocr = True
                except Exception as exc:  # pragma: no cover - depends on local tesseract
                    log.warning("OCR failed on %s page %d: %s", path.name, i, exc)
            elif len(text) < 20 and ocr:
                log.warning(
                    "%s page %d has no text layer and tesseract is not installed; "
                    "install the [ocr] extra plus tesseract-ocr-fra/ara to read scans.",
                    path.name,
                    i,
                )
            pages.append(Page(number=i, text=text, ocr=did_ocr))
    return pages


def load_text(path: Path) -> list[Page]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Page(number=1, text=text.strip())]


def load_document(path: str | Path, ocr: bool = True) -> list[Page]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    if ext == ".pdf":
        return load_pdf(path, ocr=ocr)
    return load_text(path)
