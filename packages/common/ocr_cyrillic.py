# START_MODULE_CONTRACT
# PURPOSE: Кириллический rec-движок PP-OCRv5 (Apache-2.0) поверх ONNXRuntime с собственным CTC-декодером
# SCOPE: _load_chars, ctc_decode, resize_norm_img, CyrillicRecognizer, resolve_cyrillic, run_cyrillic
# DEPENDS: numpy, cv2, onnxruntime, packages.common.config, packages.common.logging, rapidocr_onnxruntime
# LINKS: M-OCR
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ctc_decode - жадное CTC-декодирование вероятностей в строку и скор
# resize_norm_img - препроцессинг кропа: пропорциональный ресайз до 48x320, padding, нормализация
# CyrillicRecognizer - ONNX-распознаватель русского текста с собственным словарём
# resolve_cyrillic - доступен ли кириллический движок по конфигу
# run_cyrillic - полный проход детекция+классификация угла+кириллический rec
# END_MODULE_MAP
"""Кириллический OCR через PP-OCRv5 cyrillic rec.

Почему не через RapidOCR напрямую: в rapidocr_onnxruntime 1.2.3 словарь
распознавателя читается из ключа ``Rec.keys_path``, который нельзя передать
через kwargs (``rec_keys_path`` попадает в конфиг под другим именем), а
метаданных со словарём в PP-OCRv5 ONNX нет — RapidOCR поднял бы свой
китайский словарь и на русском выдал бы мусор.

Поэтому rec-стадия выполняется своим кодом: детекция и классификация угла
берутся из RapidOCR, распознавание — из PP-OCRv5 cyrillic ONNX с жадным
CTC-декодированием по словарю из ``inference.yml``
(852 выхода = blank + 850 символов + space).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np

from packages.common.config import settings
from packages.common.logging import log_marker

_REC_IMG_SHAPE = (3, 48, 320)

__all__ = [
    "CyrillicRecognizer",
    "ctc_decode",
    "resize_norm_img",
    "resolve_cyrillic",
    "run_cyrillic",
]


def _load_chars(dict_path: str | Path) -> list[str]:
    """Читает словарь символов: по одному на строку, как в inference.yml."""
    lines = Path(dict_path).read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line != ""]


def resize_norm_img(img: np.ndarray, img_shape: tuple[int, ...] = _REC_IMG_SHAPE) -> np.ndarray:
    """Пропорциональный ресайз кропа с padding и нормализацией (как в PP-OCR)."""
    img_c, img_h, img_w = img_shape
    h, w = img.shape[:2]
    ratio = min(img_w / max(w, 1), img_h / max(h, 1))
    resized_w = max(1, min(img_w, int(np.ceil(w * ratio))))
    resized = cv2.resize(img, (resized_w, img_h))
    padded = np.zeros((img_h, img_w, img_c), dtype=np.float32)
    padded[:, :resized_w, :] = resized.astype(np.float32)
    padded = padded.transpose((2, 0, 1)) / 255.0
    padded -= 0.5
    padded /= 0.5
    return padded


def ctc_decode(probs: np.ndarray, chars: list[str]) -> tuple[str, float]:
    """Жадное CTC-декодирование: argmax по времени, схлопывание повторов, blank=0."""
    indices = probs.argmax(axis=1)
    scores = probs.max(axis=1)
    text: list[str] = []
    picked: list[float] = []
    previous = -1
    for index, score in zip(indices.tolist(), scores.tolist()):
        if index != previous and index != 0:
            text.append(chars[index - 1] if 1 <= index <= len(chars) else "")
            picked.append(score)
        previous = index
    confidence = float(np.mean(picked)) if picked else 0.0
    return "".join(text), confidence


class CyrillicRecognizer:
    """ONNX-распознаватель русского текста со своим словарём и CTC-декодером."""

    def __init__(self, model_path: str | Path, dict_path: str | Path,
                 img_shape: tuple[int, ...] = _REC_IMG_SHAPE) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        self.chars = _load_chars(dict_path)
        self.img_shape = img_shape
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        expected = len(self.chars) + 2
        shape = self.session.get_outputs()[0].shape
        actual = shape[-1] if isinstance(shape[-1], int) else expected
        if actual != expected:
            raise ValueError(f"словарь не совпадает с моделью: {expected} != {actual}")

    def __call__(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Распознаёт список кропов, возвращая (текст, уверенность)."""
        if not crops:
            return []
        results: list[tuple[str, float]] = []
        for crop in crops:
            batch = resize_norm_img(crop, self.img_shape)[None, ...]
            probs = self.session.run([self.output_name], {self.input_name: batch})[0][0]
            results.append(ctc_decode(probs, self.chars))
        return results


def resolve_cyrillic() -> tuple[str, str] | None:
    """Возвращает (модель, словарь), если кириллический движок настроен и файлы на месте."""
    model = str(getattr(settings, "ocr_rec_model", "") or "")
    dict_path = str(getattr(settings, "ocr_rec_keys", "") or "")
    if not model or not dict_path:
        return None
    if not Path(model).is_file() or not Path(dict_path).is_file():
        log_marker("M-OCR", "cyrillic", "SKIP", reason="files-missing", model=model)
        return None
    return model, dict_path


def _build_detector() -> tuple[object, object]:
    """Собирает детектор и классификатор угла из штатной поставки RapidOCR."""
    utils = importlib.import_module("rapidocr_onnxruntime.utils")

    root = Path(importlib.import_module("rapidocr_onnxruntime").__file__).parent
    config = utils.concat_model_path(utils.read_yaml(str(root / "config.yaml")))
    detector = getattr(importlib.import_module("rapidocr_onnxruntime.ch_ppocr_v3_det"),
                       "TextDetector")(config["Det"])
    classifier = getattr(importlib.import_module("rapidocr_onnxruntime.ch_ppocr_v2_cls"),
                         "TextClassifier")(config["Cls"])
    return detector, classifier


_cache: dict[str, object] = {}


def run_cyrillic(image: np.ndarray, max_side: int = 768) -> list[tuple[list, str, float]]:
    """Полный проход: детекция боксов, выравнивание угла, кириллический rec.

    Returns:
        Список (box, text, score) по найденным строкам.
    """
    from packages.common.images import resize_max_side  # noqa: PLC0415

    if "det" not in _cache:
        detector, classifier = _build_detector()
        _cache["det"] = detector
        _cache["cls"] = classifier
    resolve = resolve_cyrillic()
    if resolve is None:
        return []
    if "rec" not in _cache:
        _cache["rec"] = CyrillicRecognizer(resolve[0], resolve[1])

    prepared = resize_max_side(image, max_side)
    boxes, _ = _cache["det"](prepared)  # type: ignore[operator]
    if boxes is None or len(boxes) < 1:
        return []
    crops = []
    for box in boxes:
        xs, ys = box[:, 0], box[:, 1]
        x0, x1 = int(max(0, xs.min())), int(min(prepared.shape[1], xs.max()))
        y0, y1 = int(max(0, ys.min())), int(min(prepared.shape[0], ys.max()))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        crops.append(prepared[y0:y1, x0:x1])
    if not crops:
        return []
    corrected, _, _ = _cache["cls"](crops)  # type: ignore[operator]
    texts = _cache["rec"](corrected)  # type: ignore[operator]
    return [(box, text, score) for box, (text, score) in zip(boxes, texts)]
