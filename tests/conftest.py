from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from dossier.auth import User, UserStore, set_user_store
from dossier.config import Settings
from dossier.embeddings import HashEmbedder
from dossier.llm import StubLLM
from dossier.models import DocumentMetadata
from dossier.pipeline import RagPipeline

FR_POLICY = """Politique de congés annuels

Article 1 - Droit aux congés. Chaque salarié bénéficie de 22 jours ouvrables de congé annuel payé après une année de service continu.

Article 2 - Demande. La demande de congé doit être déposée au moins 15 jours avant la date de départ via le portail RH.

Article 3 - Report. Le report des congés non pris est limité à 5 jours et doit être approuvé par le responsable hiérarchique.
"""

AR_POLICY = """سياسة العطل السنوية

المادة 1 - الحق في العطلة. يستفيد كل أجير من 22 يوم عمل من العطلة السنوية المؤدى عنها بعد سنة من الخدمة المتواصلة.

المادة 2 - الطلب. يجب تقديم طلب العطلة قبل 15 يوما على الأقل من تاريخ المغادرة عبر بوابة الموارد البشرية.
"""

INVOICE = """FACTURE N° F-2025-014
Fournisseur : Atlas Bureautique SARL - ICE 001234567000089
Date : 12/03/2025
Client : Maghreb Industries SA
Désignation : Fourniture de 20 chaises de bureau ergonomiques
Montant HT : 24 000,00 MAD
TVA 20% : 4 800,00 MAD
Montant TTC : 28 800,00 MAD
Conditions de paiement : 60 jours fin de mois.
"""


@pytest.fixture
def users(tmp_path: Path) -> dict[str, User]:
    store = UserStore(tmp_path / "missing.yaml")
    u = {
        "admin": User("admin-key", "Admin", "admin", []),
        "hr": User("hr-key", "HR", "analyst", ["rh"]),
        "finance": User("fin-key", "Finance", "analyst", ["finance"]),
    }
    for user in u.values():
        store.add(user)
    set_user_store(store)
    return u


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_backend="stub",
        embedding_backend="hash",
        qdrant_url="",
        qdrant_path=str(tmp_path / "qdrant"),
        data_dir=str(tmp_path / "data"),
        users_file=str(tmp_path / "missing.yaml"),
        min_dense_score=0.15,
        cache_similarity_threshold=0.9,
        chunk_size=300,
        chunk_overlap=40,
    )


@pytest.fixture
def pipeline(settings: Settings, tmp_path: Path) -> RagPipeline:
    p = RagPipeline(settings=settings, embedder=HashEmbedder(), llm=StubLLM())
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "conges_fr.txt").write_text(FR_POLICY, encoding="utf-8")
    (docs / "conges_ar.txt").write_text(AR_POLICY, encoding="utf-8")
    (docs / "facture_F-2025-014.txt").write_text(INVOICE, encoding="utf-8")
    p.ingest_file(docs / "conges_fr.txt", DocumentMetadata(department="rh", doc_type="policy", doc_date=date(2024, 1, 1)))
    p.ingest_file(docs / "conges_ar.txt", DocumentMetadata(department="rh", doc_type="policy"))
    p.ingest_file(
        docs / "facture_F-2025-014.txt",
        DocumentMetadata(department="finance", doc_type="invoice", doc_date=date(2025, 3, 12), tags=["atlas"]),
    )
    yield p
    p.close()
