from datetime import date

from dossier.models import RetrievalFilters


def test_ingest_registers_documents(pipeline):
    docs = pipeline.list_documents()
    assert len(docs) == 3
    langs = {d.source: d.language for d in docs}
    assert langs["conges_ar.txt"] == "ar"
    assert langs["conges_fr.txt"] == "fr"
    assert pipeline.store.count() == sum(d.chunks for d in docs)


def test_answer_is_grounded_and_cited(pipeline, users):
    r = pipeline.ask("Combien de jours de congé annuel payé ?", users["admin"])
    assert r.found
    assert "22 jours" in r.answer
    assert r.citations and r.citations[0].source == "conges_fr.txt"
    assert r.grounding_score >= 0.5
    assert not r.cache_hit


def test_arabic_question_hits_arabic_policy(pipeline, users):
    r = pipeline.ask("كم عدد أيام العطلة السنوية؟", users["hr"])
    assert r.found
    assert r.language == "ar"
    assert any(c.source == "conges_ar.txt" for c in r.citations)


def test_bm25_catches_exact_invoice_number(pipeline, users):
    r = pipeline.ask("Montant TTC de la facture F-2025-014", users["finance"])
    assert r.found
    assert "28 800,00" in r.answer
    assert any(x["bm25_score"] > 0 for x in r.retrieved)


def test_refuses_when_nothing_relevant(pipeline, users):
    r = pipeline.ask("Quelle est la capitale de l'Australie ?", users["admin"])
    assert not r.found
    assert r.citations == []
    assert "pas trouvé" in r.answer


def test_refusal_in_arabic(pipeline, users):
    r = pipeline.ask("ما هي عاصمة أستراليا؟", users["admin"])
    assert not r.found
    assert "لم أجد" in r.answer


def test_rbac_hides_other_departments(pipeline, users):
    r = pipeline.ask("Montant TTC de la facture F-2025-014", users["hr"])
    assert not r.found
    assert all(x["department"] == "rh" for x in r.retrieved)


def test_metadata_filters(pipeline, users):
    r = pipeline.ask(
        "Combien de jours de congé annuel payé ?",
        users["admin"],
        filters=RetrievalFilters(doc_type="invoice"),
    )
    assert all(x["source"].startswith("facture") for x in r.retrieved)

    r = pipeline.ask(
        "facture chaises de bureau",
        users["admin"],
        filters=RetrievalFilters(date_from=date(2025, 1, 1), date_to=date(2025, 12, 31)),
    )
    assert all(x["source"].startswith("facture") for x in r.retrieved)

    r = pipeline.ask("facture chaises de bureau", users["admin"], filters=RetrievalFilters(tags=["atlas"]))
    assert r.retrieved and all(x["source"].startswith("facture") for x in r.retrieved)


def test_semantic_cache_hits_for_near_duplicate(pipeline, users):
    q1 = "Combien de jours de congé annuel payé ?"
    first = pipeline.ask(q1, users["admin"])
    assert not first.cache_hit
    second = pipeline.ask("Combien de jours de congé annuel payé ??", users["admin"])
    assert second.cache_hit
    assert second.answer == first.answer
    assert pipeline.cache.stats()["hits"] == 1


def test_cache_is_scoped_per_user(pipeline, users):
    q = "Combien de jours de congé annuel payé ?"
    pipeline.ask(q, users["admin"])
    r = pipeline.ask(q, users["finance"])  # finance cannot see HR docs
    assert not r.cache_hit
    assert not r.found


def test_delete_document_invalidates_cache(pipeline, users):
    q = "Montant TTC de la facture F-2025-014"
    first = pipeline.ask(q, users["admin"])
    assert first.found
    doc_id = first.citations[0].doc_id
    assert pipeline.delete_document(doc_id)
    again = pipeline.ask(q, users["admin"])
    assert not again.cache_hit
    assert not again.found
