# Dossier

Un assistant documentaire interne pour entreprises au Maroc, en français et en arabe.
Projet personnel de 5ème année MIAGE.

On dépose des documents métier (factures, politiques RH, contrats, procédures), on pose une
question dans l'une des deux langues, et on obtient une réponse courte avec la citation du
document et de la page. Si l'information n'est pas dans les documents, le système le dit
au lieu d'inventer.

Je voulais un projet qui ressemble à ce qu'une DSI marocaine demanderait vraiment, pas un
chatbot de démo : des documents dans deux langues, des droits d'accès par département, et
surtout une façon de mesurer si le modèle hallucine.

## Comment ça marche

```
question ──► cache sémantique ──(hit)──► réponse en quelques ms
                 │ (miss)
                 ▼
   recherche dense (Qdrant, embeddings multilingues)   ┐
   + filtre droits d'accès + filtres métier            ├──► fusion RRF ──► top-k passages
   recherche BM25 (numéros de facture, ICE, montants)  ┘            │
                                                                    ▼
                                            score trop faible ? ──► refus honnête
                                                                    │
                                                                    ▼
                          LLM local (Ollama) : "réponds uniquement avec les SOURCES,
                          cite [n] à chaque phrase, sinon réponds NOT_FOUND"
                                                                    │
                                                                    ▼
                          vérification des citations + score d'ancrage ──► réponse
```

Quelques choix que je peux défendre :

- **Recherche hybride.** Les embeddings comprennent "quand doit-on payer" pour retrouver
  "délai de paiement", mais ils confondent facilement `F-2025-007` et `F-2025-001`. BM25
  règle le second cas. Je fusionne les deux classements avec Reciprocal Rank Fusion pour
  ne pas avoir de poids à régler à la main.
- **Les filtres s'appliquent pendant la recherche, pas après.** Filtrer le top-k après coup
  peut renvoyer zéro résultat à un utilisateur restreint et fuite des informations de
  classement. Qdrant filtre sur les métadonnées pendant la recherche ANN.
- **Le cache sémantique est dans Qdrant.** Une dépendance de moins que Redis, et une
  recherche de question similaire est de toute façon une recherche vectorielle. Chaque
  entrée est liée au périmètre d'accès de l'utilisateur, sinon un utilisateur RH pourrait
  récupérer la réponse mise en cache par la finance.
- **Le contrôle des hallucinations est en couches.** Seuil de pertinence avant d'appeler
  le modèle, prompt restreint aux sources avec citations obligatoires, `NOT_FOUND`
  autorisé, et une vérification lexicale après génération. Une réponse sans citation
  valide est traitée comme un refus.
- **Trois stratégies de découpage** (fixe, récursive, sémantique) avec un script de
  comparaison, parce que c'est la question que tout le monde pose sur un RAG et que je
  voulais une réponse chiffrée plutôt qu'une opinion.

## Résultats

Jeu de 40 questions (français et arabe, dont 6 sans réponse dans le corpus) sur un corpus
synthétique de 19 documents. Modèle de chat qwen3.5 4B via Ollama, embeddings
multilingual-e5-small sur CPU.

| Mesure | Valeur |
|---|---|
| Document attendu dans le top-5 | 100 % |
| Questions avec réponse effectivement répondues | 34 / 34 |
| Questions sans réponse effectivement refusées | 6 / 6 |
| Réponses contenant la bonne valeur | 94 % |
| Citations pointant vers le bon document | 98 % |
| Latence médiane à froid | 3,6 s |
| Latence médiane sur cache sémantique | 29 ms |

Les 8 questions en arabe ont été répondues avec des citations valides, y compris celles
dont la réponse n'existait qu'en français. Le rapport complet est dans
[eval/results.md](eval/results.md) et la comparaison des découpages dans
[eval/chunking.md](eval/chunking.md). Sur ce petit corpus les trois stratégies
retrouvent le bon document ; le découpage récursif à 800 caractères produit 40 % de
passages en moins que le découpage fixe pour le même rang moyen, donc moins de tokens par
appel, c'est pourquoi il est par défaut.

## Ce que j'ai appris en le construisant

- Une carte graphique de 8 Go ne fait pas tenir un modèle de chat 4B et bge-m3 en même
  temps. Ollama les échange à chaque requête (10 à 17 s) et le runner d'embeddings
  plante parfois avec une erreur 500. J'ai ajouté des réessais avec backoff, puis
  déplacé les embeddings sur CPU avec sentence-transformers. C'est la configuration
  utilisée pour les résultats ci-dessus.
- Les petits modèles ne respectent pas toujours le token `NOT_FOUND` : ils écrivent
  "لا توجد إجابة في المصادر" en prose. D'où la règle : pas de citation, pas de réponse.
- Envoyer de l'arabe avec `curl` depuis Git Bash sous Windows corrompt l'encodage. Les
  tests passent par `requests` en Python.
- Le Qdrant embarqué n'accepte qu'un seul processus. Il faut arrêter l'API avant de
  lancer les scripts d'ingestion, ou ingérer via l'API avec `--api`.
- Générer des PDF en arabe avec fpdf2 demande une police et un moteur de mise en forme
  bidirectionnelle ; la politique RH en arabe est donc en Markdown, ce que l'ingestion
  gère de la même façon.

## Lancer le projet

Installation :

```bash
pip install -e ".[dev,data,ui,embeddings]"
cp .env.example .env
```

Avec Ollama (configuration recommandée, celle des résultats) :

```bash
ollama pull qwen3.5:4b
# dans .env : LLM_BACKEND=ollama, OLLAMA_MODEL=qwen3.5:4b, EMBEDDING_BACKEND=sentence-transformers
python scripts/generate_dataset.py     # corpus synthétique -> data/raw
python scripts/ingest.py               # indexation
uvicorn dossier.api:app --port 8000    # API, docs sur http://localhost:8000
streamlit run ui/app.py                # interface sur http://localhost:8501
```

Clés de démo dans [config/users.yaml](config/users.yaml) : `admin-key` voit tout,
`finance-key`, `hr-key` et `procurement-key` ne voient que leur département.

Sans aucun modèle (tests, CI) : `LLM_BACKEND=stub EMBEDDING_BACKEND=hash MIN_DENSE_SCORE=0.15`.
Le stub extrait des phrases au lieu de générer ; c'est un test de plomberie, pas de qualité.

Tout en conteneurs : `docker compose up --build` lance Qdrant, Ollama, l'API et l'interface.

Évaluation et comparaison des découpages (API arrêtée) :

```bash
python scripts/evaluate.py
python scripts/compare_chunking.py --sizes 400 800
```

## Exemple d'appel

```bash
curl -X POST http://localhost:8000/ask -H "X-API-Key: finance-key" -H "Content-Type: application/json" \
  -d '{"question": "Quel est le montant TTC de la facture F-2025-003 ?", "filters": {"doc_type": "invoice"}}'
```

```json
{
  "answer": "Le montant TTC de la facture F-2025-003 est de 20 400,00 MAD [1].",
  "found": true,
  "citations": [{"n": 1, "source": "facture_F-2025-003.pdf", "page": 1, "excerpt": "FACTURE N° F-2025-003 ..."}],
  "grounding_score": 1.0,
  "low_confidence": false,
  "cache_hit": false,
  "latency_ms": 3591
}
```

Un utilisateur RH qui pose la même question obtient un refus : la facture n'est jamais
récupérée pour lui, et la réponse en cache de la finance ne lui est pas servie.

## Organisation du code

```
src/dossier/
  api.py          routes FastAPI (upload, ask, documents, cache)
  pipeline.py     ingestion et réponse, assemble le reste
  ingestion/      lecture PDF/OCR/texte, détection de langue, découpage
  embeddings.py   sentence-transformers, Ollama ou hash (tests)
  vectorstore.py  Qdrant avec filtres métier et droits d'accès
  bm25_index.py   BM25 avec normalisation FR/AR (accents, diacritiques, ال)
  retrieval.py    fusion RRF
  generation.py   prompt, citations, score d'ancrage
  cache.py        cache sémantique par périmètre d'accès
  auth.py         clés API, rôles, départements
scripts/          génération du corpus, ingestion, évaluation, comparaison des découpages
eval/             40 questions et les rapports
tests/            26 tests (découpage, pipeline, droits, cache, API)
```

## Pistes

- Juge LLM pour la fidélité (RAGAS) en plus du score lexical.
- Reranker (bge-reranker-v2-m3) entre la fusion et la génération.
- Extraction structurée des champs de facture (fournisseur, ICE, HT/TVA/TTC).
- SSO à la place des clés API, et journal d'audit des questions.
- Questions en darija avec un modèle adapté (Atlas-Chat).

Le corpus est entièrement synthétique (société fictive "Maghreb Industries SA").
Licence MIT.
