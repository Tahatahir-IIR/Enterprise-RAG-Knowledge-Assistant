# Chunking strategy comparison

| strategy | size | overlap | chunks | avg_chars | hit@5 | mrr |
|---|---|---|---|---|---|---|
| fixed | 400 | 100 | 64 | 311 | 1.0 | 0.966 |
| fixed | 800 | 100 | 38 | 436 | 1.0 | 0.963 |
| recursive | 400 | 100 | 62 | 283 | 1.0 | 0.951 |
| recursive | 800 | 100 | 24 | 657 | 1.0 | 0.963 |
| semantic | 400 | 100 | 63 | 249 | 1.0 | 0.966 |
| semantic | 800 | 100 | 53 | 296 | 1.0 | 0.966 |