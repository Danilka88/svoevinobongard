#!/usr/bin/env python3
# START_MODULE_CONTRACT
# PURPOSE: Обучение контрастивной проекционной головы (InfoNCE с hard negatives) поверх замороженного энкодера
# SCOPE: parse_args, main
# DEPENDS: numpy, torch, json, packages.common.logging, services.retrieval.app.head
# LINKS: M-RETRIEVAL, M-EVAL
# ROLE: SCRIPT
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# parse_args - разбор аргументов CLI
# main - обучение головы на кэшированных эмбеддингах и запись весов и отчёта
# END_MODULE_MAP
"""Домен-адаптация: обучение проекционной головы.

Обучается только голова; энкодер заморожен, его эмбеддинги закэшированы
``synthetic_pairs.py``, поэтому эпоха занимает секунды даже на большой модели.

Loss: InfoNCE. Для каждого anchor (случайный из аугментаций пары) позитивом
служит чистый эталон того же вина, негативами — эталоны других вин батча плюс
hard negatives (та же винодельня или сорт).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from packages.common.logging import log_marker  # noqa: E402
from services.retrieval.app.head import ProjectionHead, save_head  # noqa: E402

__all__ = ["parse_args", "main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы CLI."""
    parser = argparse.ArgumentParser(description="Обучение проекционной головы (InfoNCE)")
    parser.add_argument("--pairs", default=str(_ROOT / "data" / "train" / "pairs.jsonl"))
    parser.add_argument("--embeddings", default=str(_ROOT / "data" / "train" / "embeddings.npy"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-seconds", type=float, default=1200.0, help="timebox обучения")
    parser.add_argument("--out", default=str(_ROOT / "models" / "head.pt"))
    parser.add_argument("--report", default=str(_ROOT / "reports" / "train_head.json"))
    return parser.parse_args(argv)


def _load_pairs(path: Path) -> list[dict]:
    """Читает pairs.jsonl."""
    items: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return [item for item in items if item.get("anchors") and "positive" in item]


def _train_epoch(
    head: ProjectionHead,
    optimizer: torch.optim.Optimizer,
    embeddings: torch.Tensor,
    positives: np.ndarray,
    anchors_by_pair: list[list[int]],
    hard_buckets: list[np.ndarray],
    batch: int,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Одна эпоха InfoNCE.

    Returns:
        (средний loss, accuracy@1 внутри батча).
    """
    order = rng.permutation(len(positives))
    total_loss = 0.0
    correct = 0
    seen = 0
    batches = 0
    criterion = nn.CrossEntropyLoss()

    for start in range(0, len(order), batch):
        idx = order[start:start + batch]
        if len(idx) < 4:
            continue

        anchor_rows = [int(rng.choice(anchors_by_pair[int(i)])) for i in idx]
        anchor_idx = torch.tensor(anchor_rows, dtype=torch.long, device=embeddings.device)
        pos_idx = torch.tensor([int(positives[int(i)]) for i in idx],
                               dtype=torch.long, device=embeddings.device)

        anchor_vec = head(embeddings[anchor_idx])
        gallery = [head(embeddings[pos_idx])]
        for bucket in hard_buckets:
            selected = [int(bucket[int(i)]) for i in idx if int(bucket[int(i)]) >= 0]
            if selected:
                rows = torch.tensor(selected, dtype=torch.long, device=embeddings.device)
                gallery.append(head(embeddings[rows]))
        gallery_vec = torch.cat(gallery, dim=0)

        logits = (anchor_vec @ gallery_vec.t()) / temperature
        labels = torch.arange(len(idx), dtype=torch.long, device=embeddings.device)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        seen += len(idx)
        batches += 1

    if batches == 0 or seen == 0:
        return 0.0, 0.0
    return total_loss / batches, correct / float(seen)


def main(argv: list[str] | None = None) -> int:
    """Обучает голову и сохраняет веса с отчётом."""
    args = parse_args(argv)
    pairs = _load_pairs(Path(args.pairs))
    embeddings_np = np.load(args.embeddings).astype(np.float32)
    if not pairs:
        print("ERROR: no pairs found", file=sys.stderr)
        return 2

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embeddings = torch.from_numpy(embeddings_np).to(device)

    positives = np.array([int(item["positive"]) for item in pairs], dtype=np.int64)
    anchors_by_pair = [[int(index) for index in item["anchors"]] for item in pairs]
    max_hard = max(len(item["hard_negatives"]) for item in pairs)
    hard_buckets: list[np.ndarray] = []
    for slot in range(max_hard):
        hard_buckets.append(np.array(
            [int(item["hard_negatives"][slot]) if slot < len(item["hard_negatives"]) else -1
             for item in pairs],
            dtype=np.int64,
        ))
    total_anchors = sum(len(item) for item in anchors_by_pair)

    head = ProjectionHead(int(embeddings_np.shape[1]), args.hidden, args.out_dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    log_marker("M-RETRIEVAL", "train_head", "START", pairs=len(pairs), anchors=total_anchors,
               dim=int(embeddings_np.shape[1]), device=device)
    started = time.perf_counter()
    history: list[dict] = []
    stopped_reason = "epochs-exhausted"

    for epoch in range(1, args.epochs + 1):
        loss, accuracy = _train_epoch(head, optimizer, embeddings, positives,
                                      anchors_by_pair, hard_buckets,
                                      args.batch, args.temperature, rng)
        elapsed = time.perf_counter() - started
        history.append({"epoch": epoch, "loss": round(loss, 5),
                        "batch_acc@1": round(accuracy, 4), "elapsed_s": round(elapsed, 1)})
        print(f"  epoch {epoch:2d} loss={loss:.4f} batch_acc@1={accuracy:.3f} ({elapsed:.1f}s)",
              file=sys.stderr)
        if elapsed > args.max_seconds:
            stopped_reason = f"timebox-{args.max_seconds:.0f}s"
            break

    train_seconds = time.perf_counter() - started
    metrics = {
        "pairs": len(pairs),
        "anchors": total_anchors,
        "dim": int(embeddings_np.shape[1]),
        "hidden": args.hidden,
        "out_dim": args.out_dim,
        "epochs_run": len(history),
        "loss_first": history[0]["loss"] if history else 0.0,
        "loss_last": history[-1]["loss"] if history else 0.0,
        "batch_acc_first": history[0]["batch_acc@1"] if history else 0.0,
        "batch_acc_last": history[-1]["batch_acc@1"] if history else 0.0,
        "train_seconds": round(train_seconds, 1),
        "stopped_reason": stopped_reason,
        "device": device,
    }
    save_head(args.out, head, metrics)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"metrics": metrics, "history": history},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    log_marker("M-RETRIEVAL", "train_head", "DONE", epochs=len(history),
               loss_last=metrics["loss_last"], seconds=round(train_seconds, 1))

    print("=== TRAIN REPORT ===")
    print(f"pairs/anchors:  {metrics['pairs']} / {metrics['anchors']} (dim {metrics['dim']})")
    print(f"head:           {metrics['dim']} -> {args.hidden} -> {args.out_dim}")
    print(f"epochs run:     {metrics['epochs_run']} ({stopped_reason})")
    print(f"loss:           {metrics['loss_first']:.4f} -> {metrics['loss_last']:.4f}")
    print(f"batch acc@1:    {metrics['batch_acc_first']:.3f} -> {metrics['batch_acc_last']:.3f}")
    print(f"train seconds:  {train_seconds:.1f}")
    print(f"device:         {device}")
    print(f"head:           {args.out}")
    print(f"report:         {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
