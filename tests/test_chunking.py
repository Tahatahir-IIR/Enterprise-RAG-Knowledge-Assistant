from dossier.embeddings import HashEmbedder
from dossier.ingestion import build_chunker, chunk_pages, detect_language
from dossier.ingestion.loaders import Page

TEXT = ("Première phrase sur les congés. " * 10 + "\n\n" + "Deuxième sujet sur les factures et la TVA. " * 10).strip()


def test_fixed_chunker_respects_size_and_overlap():
    chunks = build_chunker("fixed", size=100, overlap=20).split(TEXT)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) > 1


def test_recursive_chunker_prefers_paragraph_breaks():
    chunks = build_chunker("recursive", size=400, overlap=0).split(TEXT)
    assert len(chunks) >= 2
    assert not any("congés" in c and "factures" in c for c in chunks)


def test_semantic_chunker_splits_on_topic_change():
    chunks = build_chunker("semantic", size=2000, embedder=HashEmbedder()).split(TEXT)
    assert len(chunks) >= 2
    assert "congés" in chunks[0]


def test_chunk_pages_keeps_page_numbers():
    pages = [Page(1, "a b c. " * 50), Page(2, "x y z. " * 50)]
    pieces = chunk_pages(pages, build_chunker("fixed", size=80, overlap=0))
    assert {p for p, _ in pieces} == {1, 2}


def test_language_detection():
    assert detect_language("Quel est le montant TTC ?") == "fr"
    assert detect_language("ما هو المبلغ الإجمالي؟") == "ar"
    assert detect_language("") == "fr"
