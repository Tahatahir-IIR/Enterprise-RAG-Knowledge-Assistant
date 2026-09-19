import pytest
from fastapi.testclient import TestClient

from dossier import api


@pytest.fixture
def client(pipeline, users):
    api.set_pipeline(pipeline)
    with TestClient(api.app) as c:
        yield c


def test_requires_api_key(client):
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "nope"}).status_code == 401


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 3


def test_list_documents_is_filtered_by_role(client):
    admin = client.get("/documents", headers={"X-API-Key": "admin-key"}).json()
    hr = client.get("/documents", headers={"X-API-Key": "hr-key"}).json()
    assert len(admin) == 3 and len(hr) == 2
    assert all(d["department"] == "rh" for d in hr)


def test_ask_endpoint(client):
    r = client.post(
        "/ask",
        json={"question": "Combien de jours de congé annuel payé ?"},
        headers={"X-API-Key": "hr-key"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["found"] and body["citations"][0]["page"] == 1


def test_ask_forbidden_department_filter(client):
    r = client.post(
        "/ask",
        json={"question": "x y", "filters": {"department": "finance"}},
        headers={"X-API-Key": "hr-key"},
    )
    assert r.status_code == 403


def test_upload_and_delete(client, tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Procédure achats\n\nTout bon de commande supérieur à 50 000 MAD requiert deux signatures.", encoding="utf-8")
    with f.open("rb") as fh:
        r = client.post(
            "/documents",
            files={"file": ("note.md", fh, "text/markdown")},
            data={"department": "finance", "doc_type": "procedure", "tags": "achats, signature"},
            headers={"X-API-Key": "fin-key"},
        )
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["tags"] == ["achats", "signature"]

    ask = client.post("/ask", json={"question": "Combien de signatures pour un bon de commande supérieur à 50 000 MAD ?"}, headers={"X-API-Key": "fin-key"}).json()
    assert ask["found"] and "deux signatures" in ask["answer"]

    assert client.delete(f"/documents/{doc['doc_id']}", headers={"X-API-Key": "hr-key"}).status_code == 403
    assert client.delete(f"/documents/{doc['doc_id']}", headers={"X-API-Key": "fin-key"}).status_code == 204


def test_upload_to_forbidden_department(client, tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")
    with f.open("rb") as fh:
        r = client.post(
            "/documents",
            files={"file": ("x.txt", fh, "text/plain")},
            data={"department": "finance", "doc_type": "other"},
            headers={"X-API-Key": "hr-key"},
        )
    assert r.status_code == 403


def test_cache_clear_is_admin_only(client):
    assert client.delete("/cache", headers={"X-API-Key": "hr-key"}).status_code == 403
    assert client.delete("/cache", headers={"X-API-Key": "admin-key"}).status_code == 204
