from dossier.generation import generate
from dossier.llm import LLM
from dossier.models import Chunk, RetrievedChunk


class FixedLLM(LLM):
    def __init__(self, text):
        self.text = text

    def complete(self, system, user, temperature=0.0):
        return self.text


def _chunks():
    c = Chunk(chunk_id="d:0", doc_id="d", source="x.pdf", page=1, chunk_index=0, text="Le plafond est 150 MAD par repas.",
              department="finance", doc_type="procedure", language="fr")
    return [RetrievedChunk(chunk=c)]


def test_uncited_prose_refusal_is_treated_as_not_found():
    g = generate(FixedLLM("لا توجد إجابة في المصادر المقدمة."), "q", _chunks())
    assert not g.found and g.citations == []


def test_cited_answer_is_found_and_grounded():
    g = generate(FixedLLM("Le plafond est 150 MAD par repas [1]."), "q", _chunks())
    assert g.found and g.citations[0].page == 1 and g.grounding_score == 1.0
