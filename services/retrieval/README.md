# services/retrieval

Поиск по каталогу (Tier-1).

- Энкодер-плагин: `siglip2 | dinov2 | clip` (инференс MPS/ONNX).
- Индекс: pgvector (prod) / FAISS (dev); несколько эмбеддингов на позицию.
- Top-K по косинусу.
- Геоверификация: SuperPoint + LightGlue + RANSAC → скор + margin.

Интерфейс (план): `POST /search` (embeddings|image) → Top-K + скоры.

Статус: скелет (M0), реализация — M1–M2.
