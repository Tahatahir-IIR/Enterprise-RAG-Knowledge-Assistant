"""Streamlit front-end for the Dossier API.

  streamlit run ui/app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="Enterprise RAG Knowledge Assistant", page_icon="📁", layout="wide")

with st.sidebar:
    st.title("📁 Enterprise RAG Knowledge Assistant")
    st.caption("Assistant documentaire interne · FR / AR")
    api_url = st.text_input("API URL", os.getenv("API_URL", "http://localhost:8000"))
    api_key = st.text_input("Clé API", os.getenv("API_KEY", "admin-key"), type="password")
    st.caption("Clés de démo : admin-key · finance-key · hr-key · procurement-key")
    try:
        h = requests.get(f"{api_url}/health", timeout=5).json()
        st.success(f"{h['documents']} documents · {h['chunks']} chunks · LLM {h['llm']}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API injoignable : {exc}")

HEADERS = {"X-API-Key": api_key}
tab_ask, tab_docs, tab_cache = st.tabs(["Poser une question", "Documents", "Cache"])

with tab_ask:
    question = st.text_area("Question (français ou عربي)", placeholder="Quel est le montant TTC de la facture F-2025-003 ?", height=80)
    c1, c2, c3, c4 = st.columns(4)
    doc_type = c1.selectbox("Type", ["", "invoice", "policy", "contract", "procedure", "report", "other"])
    language = c2.selectbox("Langue", ["", "fr", "ar"])
    date_from = c3.date_input("Du", value=None)
    date_to = c4.date_input("Au", value=None)
    c5, c6 = st.columns(2)
    top_k = c5.slider("Passages récupérés", 1, 10, 5)
    use_cache = c6.toggle("Cache sémantique", value=True)

    if st.button("Demander", type="primary", disabled=not question.strip()):
        payload = {
            "question": question.strip(),
            "top_k": top_k,
            "use_cache": use_cache,
            "filters": {
                "doc_type": doc_type or None,
                "language": language or None,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None,
            },
        }
        with st.spinner("Recherche…"):
            r = requests.post(f"{api_url}/ask", json=payload, headers=HEADERS, timeout=300)
        if r.status_code != 200:
            st.error(f"{r.status_code}: {r.text}")
        else:
            body = r.json()
            badges = [f"⏱ {body['latency_ms']} ms"]
            badges.append("⚡ cache" if body["cache_hit"] else "🧠 LLM")
            if body["found"]:
                badges.append(f"🔗 ancrage {body['grounding_score']:.0%}")
                if body["low_confidence"]:
                    badges.append("⚠️ faible confiance")
            st.caption(" · ".join(badges))
            if body["found"]:
                st.markdown(f"### {body['answer']}" if len(body["answer"]) < 120 else body["answer"])
                if body["low_confidence"]:
                    st.warning("Réponse peu appuyée par les sources : vérifiez les extraits cités avant de l'utiliser.")
                with st.expander(f"Sources citées ({len(body['citations'])})", expanded=True):
                    for c in body["citations"]:
                        st.markdown(f"**[{c['n']}] {c['source']}** · page {c['page']}")
                        st.code(c["excerpt"], language=None)
            else:
                st.info(body["answer"])
            if body["retrieved"]:
                with st.expander("Passages récupérés (dense / BM25 / fusion)"):
                    st.dataframe(body["retrieved"], use_container_width=True, hide_index=True)

with tab_docs:
    st.subheader("Ajouter un document")
    with st.form("upload"):
        f = st.file_uploader("PDF, TXT ou Markdown", type=["pdf", "txt", "md"])
        c1, c2, c3 = st.columns(3)
        dep = c1.text_input("Département", "finance")
        dtype = c2.selectbox("Type de document", ["invoice", "policy", "contract", "procedure", "report", "other"])
        ddate = c3.date_input("Date du document", value=None)
        tags = st.text_input("Tags (séparés par des virgules)", "")
        if st.form_submit_button("Indexer") and f is not None:
            r = requests.post(
                f"{api_url}/documents",
                files={"file": (f.name, f.getvalue())},
                data={"department": dep, "doc_type": dtype, "doc_date": ddate.isoformat() if ddate else "", "tags": tags},
                headers=HEADERS,
                timeout=600,
            )
            if r.status_code == 201:
                d = r.json()
                st.success(f"{d['source']} indexé : {d['pages']} pages, {d['chunks']} chunks, langue {d['language']}")
            else:
                st.error(f"{r.status_code}: {r.text}")

    st.subheader("Documents accessibles")
    r = requests.get(f"{api_url}/documents", headers=HEADERS, timeout=30)
    if r.status_code == 200:
        docs = r.json()
        if docs:
            st.dataframe(
                [{k: d[k] for k in ["source", "department", "doc_type", "language", "doc_date", "pages", "chunks", "doc_id"]} for d in docs],
                use_container_width=True,
                hide_index=True,
            )
            to_delete = st.selectbox("Supprimer", [""] + [f"{d['source']} ({d['doc_id']})" for d in docs])
            if to_delete and st.button("Supprimer définitivement"):
                doc_id = to_delete.rsplit("(", 1)[1].rstrip(")")
                rr = requests.delete(f"{api_url}/documents/{doc_id}", headers=HEADERS, timeout=30)
                st.success("Supprimé") if rr.status_code == 204 else st.error(rr.text)
                st.rerun()
        else:
            st.info("Aucun document. Lancez `python scripts/generate_dataset.py` puis `python scripts/ingest.py --api ...`.")
    else:
        st.error(f"{r.status_code}: {r.text}")

with tab_cache:
    r = requests.get(f"{api_url}/cache/stats", headers=HEADERS, timeout=30)
    if r.status_code == 200:
        s = r.json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entrées", s["entries"])
        c2.metric("Hits", s["hits"])
        c3.metric("Misses", s["misses"])
        c4.metric("Taux de hit", f"{s['hit_rate']:.0%}")
        st.caption(f"Seuil de similarité : {s['threshold']}")
        if st.button("Vider le cache (admin)"):
            rr = requests.delete(f"{api_url}/cache", headers=HEADERS, timeout=30)
            st.success("Cache vidé") if rr.status_code == 204 else st.error(rr.text)
    else:
        st.error(f"{r.status_code}: {r.text}")
