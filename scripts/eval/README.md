# scripts/eval

`evaluate.py` — оценка сервиса: шлёт изображения по одному, собирает slug/top5/latency,
считает accuracy@1/@5 и macro-F1 при наличии ground truth.

```bash
python3 scripts/eval/evaluate.py \
  --images data/samples/field \
  --endpoint http://127.0.0.1:8080/v1/scan \
  --output predictions.jsonl
```

Официальный скрипт организатора (`participant_test.sh`) лежит в материалах конкурса;
наш эвалюатор повторяет его контракт (`multipart` поле `image`, ответ `{"slug": ...}`)
и дополнительно снимает `top5` и `confidence` из `/v1/scan`.

Ground truth (если есть): `--ground-truth gt.tsv` формата `query_id<TAB>slug`.
