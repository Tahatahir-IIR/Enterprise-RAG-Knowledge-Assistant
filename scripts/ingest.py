"""Ingest every document listed in a manifest.

Direct mode (default) writes to the embedded index; stop the API first because
embedded Qdrant allows one process at a time. With --api the files are uploaded
through the running server instead, which works with both embedded and server
Qdrant.

  python scripts/ingest.py
  python scripts/ingest.py --api http://localhost:8000 --api-key admin-key
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def via_api(manifest: list[dict], base: Path, api: str, api_key: str) -> None:
    import requests

    for entry in manifest:
        path = base / entry["file"]
        with path.open("rb") as fh:
            r = requests.post(
                f"{api.rstrip('/')}/documents",
                files={"file": (path.name, fh)},
                data={
                    "department": entry["department"],
                    "doc_type": entry["doc_type"],
                    "language": entry.get("language") or "",
                    "doc_date": entry.get("doc_date") or "",
                    "tags": ",".join(entry.get("tags", [])),
                },
                headers={"X-API-Key": api_key},
                timeout=600,
            )
        r.raise_for_status()
        info = r.json()
        print(f"  {info['source']:<40} {info['pages']:>2} pages {info['chunks']:>3} chunks  [{info['language']}]")


def direct(manifest: list[dict], base: Path) -> None:
    from dossier.models import DocumentMetadata
    from dossier.pipeline import RagPipeline

    p = RagPipeline()
    try:
        for entry in manifest:
            meta = DocumentMetadata(
                department=entry["department"],
                doc_type=entry["doc_type"],
                language=entry.get("language"),
                doc_date=date.fromisoformat(entry["doc_date"]) if entry.get("doc_date") else None,
                tags=entry.get("tags", []),
            )
            info = p.ingest_file(base / entry["file"], meta)
            print(f"  {info.source:<40} {info.pages:>2} pages {info.chunks:>3} chunks  [{info.language}]")
        print(f"Index now holds {len(p.documents)} documents / {p.store.count()} chunks")
    finally:
        p.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/raw/manifest.json")
    ap.add_argument("--api", default="", help="Base URL of a running API; omit for direct ingestion")
    ap.add_argument("--api-key", default="admin-key")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    print(f"Ingesting {len(manifest)} documents from {base}/")
    if args.api:
        via_api(manifest, base, args.api, args.api_key)
    else:
        direct(manifest, base)


if __name__ == "__main__":
    main()
