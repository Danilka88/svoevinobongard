# START_MODULE_CONTRACT
# PURPOSE: Проекционная голова домен-адаптации: линейная адаптация эмбеддингов замороженного энкодера
# SCOPE: ProjectionHead, save_head, load_head, apply_head, head_meta, read_head_meta, describe_head
# DEPENDS: torch, numpy, json, pathlib
# LINKS: M-RETRIEVAL
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ProjectionHead - MLP-голова с L2-нормализацией выхода
# save_head - сохранение весов и метаданных головы
# load_head - загрузка головы по метаданным индекса
# apply_head - применение головы к матрице эмбеддингов
# head_meta - метаданные головы для meta.json индекса
# read_head_meta - описание головы из meta.json индекса
# describe_head - человекочитаемое описание головы для логов
# END_MODULE_MAP
"""Домен-адаптация Tier-1.

Энкодер остаётся замороженным: обучается только проекционная голова (adapter),
которая переводит эмбеддинги из домена «эталонный вырез» в домен «полевое фото».
Голова применяется и к эталонам при сборке индекса, и к запросам при поиске —
поэтому она хранится рядом с индексом и описывается в meta.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

__all__ = ["ProjectionHead", "save_head", "load_head", "apply_head", "head_meta",
           "read_head_meta", "describe_head"]

HEAD_FILE = "head.pt"


class ProjectionHead(nn.Module):
    """MLP-голова с GELU и L2-нормализацией выхода."""

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Проецирует эмбеддинги и нормирует результат."""
        projected = self.net(vectors)
        return nn.functional.normalize(projected, dim=-1)


def head_meta(head: ProjectionHead, metrics: dict | None = None) -> dict:
    """Собирает метаданные головы для записи в meta.json индекса."""
    payload = {
        "file": HEAD_FILE,
        "in_dim": head.in_dim,
        "hidden": head.hidden,
        "out_dim": head.out_dim,
        "metrics": metrics or {},
    }
    return payload


def save_head(path: str | Path, head: ProjectionHead, metrics: dict | None = None) -> None:
    """Сохраняет веса головы и её конфигурацию."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "in_dim": head.in_dim,
            "hidden": head.hidden,
            "out_dim": head.out_dim,
            "metrics": metrics or {},
        },
        target,
    )


def load_head(path: str | Path, device: str = "cpu") -> ProjectionHead:
    """Загружает голову из файла.

    Raises:
        FileNotFoundError: если файл головы отсутствует.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"head not found: {source}")
    payload = torch.load(source, map_location=device, weights_only=False)
    head = ProjectionHead(int(payload["in_dim"]), int(payload["hidden"]), int(payload["out_dim"]))
    head.load_state_dict(payload["state_dict"])
    head.eval()
    head.to(device)
    return head


def apply_head(head: ProjectionHead | None, vectors: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Применяет голову к L2-нормированной матрице эмбеддингов."""
    if head is None or vectors.size == 0:
        return vectors
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32)).to(device)
        projected = head(tensor).detach().to("cpu").numpy().astype(np.float32)
    return projected


def read_head_meta(meta: dict) -> dict | None:
    """Возвращает описание головы из meta.json индекса, если оно есть."""
    payload = (meta or {}).get("head")
    return payload if isinstance(payload, dict) and payload.get("file") else None


def describe_head(meta: dict) -> str:
    """Человекочитаемое описание головы для логов и отчётов."""
    payload = read_head_meta(meta)
    if not payload:
        return "none"
    return f"{payload['in_dim']}->{payload['hidden']}->{payload['out_dim']}"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
