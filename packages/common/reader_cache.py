# START_MODULE_CONTRACT
# PURPOSE: Постоянный кэш чтений этикетки (26b) по хэшу фото: повторно модель не гоняется без изменений
# SCOPE: PROMPT_VERSION, DEFAULT_CACHE_DIR, image_key, cache_path, get_cached_reading, put_cached_reading,
#        clear_reader_cache
# DEPENDS: hashlib, json, pathlib, packages.common.observer
# LINKS: M-EVAL, M-COMMON
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# PROMPT_VERSION - версия промпта чтения: изменение инструкции инвалидирует кэш
# DEFAULT_CACHE_DIR - каталог кэша чтений на диске
# image_key - ключ кэша по содержимому фото (sha256) + модели + версии промпта
# cache_path - путь файла кэша для ключа
# get_cached_reading - чтение из кэша или None (с проверкой ключа)
# put_cached_reading - запись чтения в кэш
# clear_reader_cache - очистка кэша (тесты и операционная гигиена)
# END_MODULE_MAP
"""Постоянный кэш чтений этикетки.

Сильная модель (26b) читает этикетку секунды-десятки секунд и грузит GPU. Одно и то же фото
(gold/real100, демо, отладка) не должно читаться повторно, если не изменились ни фото, ни модель,
ни промпт. Ключ кэша — ``sha256(фото) + модель + версия промпта``: меняется любой элемент —
запись считается устаревшей и чтение выполняется заново. Кэш живёт на диске и переживает перезапуск.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = ["DEFAULT_CACHE_DIR", "PROMPT_VERSION", "cache_path", "clear_reader_cache",
           "get_cached_reading", "image_key", "put_cached_reading"]

# Версия инструкции чтения. Меняй при правке OBSERVER_INSTRUCTION/OBSERVER_STRUCTURED_INSTRUCTION —
# тогда старые записи кэша не будут ошибочно переиспользованы.
PROMPT_VERSION = "observer-v1"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "reader"


def image_key(image_path: str | Path, model: str, *, prompt_version: str = PROMPT_VERSION,
              extra: str = "") -> str:
    """Ключ кэша чтения: sha256 содержимого фото + модель + версия промпта.

    Хэш считается по **байтам файла**, поэтому правка фото (перекроп, другая версия) автоматически
    инвалидирует кэш. ``extra`` позволяет учесть дополнительные параметры чтения (например пайплайн
    препроцессинга), если они меняют результат.
    """
    digest = hashlib.sha256()
    with open(image_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    payload = f"{digest.hexdigest()}|{model}|{prompt_version}|{extra}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(key: str, *, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    """Путь файла кэша для ключа."""
    return Path(cache_dir) / f"{key}.json"


def get_cached_reading(key: str, *, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> dict | None:
    """Чтение из кэша или None, если записи нет или она повреждена."""
    path = cache_path(key, cache_dir=cache_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(payload, dict) or "raw_text" not in payload:
        return None
    result = dict(payload)
    result["from_cache"] = True
    return result


def put_cached_reading(key: str, reading: dict, *, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> bool:
    """Записывает чтение в кэш; возвращает False, если запись не удалась (не критично)."""
    path = cache_path(key, cache_dir=cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {k: v for k, v in dict(reading).items() if k != "from_cache"}
        record.setdefault("key", key)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except OSError:
        return False


def clear_reader_cache(*, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> int:
    """Удаляет все файлы кэша чтений; возвращает число удалённых."""
    directory = Path(cache_dir)
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.glob("*.json"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed
