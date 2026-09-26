# START_MODULE_CONTRACT
# PURPOSE: Тесты C-087 постоянного кэша чтений: ключ, хит/мисс, инвалидация по фото/модели/промпту, очистка
# SCOPE: test_image_key_stable_and_changes, test_cache_hit_and_miss, test_invalidation_by_model_and_prompt,
#        test_corrupt_cache_is_miss, test_clear_reader_cache, test_observe_reading_uses_cache
# DEPENDS: packages.common.reader_cache, scripts.eval.llm_match_eval
# LINKS: M-COMMON, M-EVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# test_image_key_stable_and_changes - ключ стабилен для одного фото и меняется при правке/модели/промпте
# test_cache_hit_and_miss - запись читается обратно, отсутствующий ключ даёт None
# test_invalidation_by_model_and_prompt - смена модели или версии промпта делает кэш-мисс
# test_corrupt_cache_is_miss - повреждённый файл трактуется как кэш-мисс
# test_clear_reader_cache - очистка удаляет все записи
# test_observe_reading_uses_cache - observe_reading берёт из кэша и не вызывает модель повторно
# END_MODULE_MAP
"""Тесты постоянного кэша чтений C-087 (без сети и моделей)."""
from __future__ import annotations

from packages.common.reader_cache import (clear_reader_cache, get_cached_reading, image_key,
                                          put_cached_reading)
from scripts.eval.llm_match_eval import observe_reading

__all__ = ["test_cache_hit_and_miss", "test_clear_reader_cache", "test_corrupt_cache_is_miss",
           "test_image_key_stable_and_changes", "test_invalidation_by_model_and_prompt",
           "test_observe_reading_uses_cache"]


def _photo(tmp_path, name: str = "photo.jpg", data: bytes = b"fake-image") -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def test_image_key_stable_and_changes(tmp_path) -> None:
    """Ключ стабилен для одного фото и меняется при изменении фото/модели/промпта."""
    photo = _photo(tmp_path)
    base = image_key(photo, "model-a")
    assert base == image_key(photo, "model-a")
    assert base != image_key(photo, "model-b")
    assert base != image_key(photo, "model-a", prompt_version="v2")
    other = _photo(tmp_path, "other.jpg", b"other-bytes")
    assert base != image_key(other, "model-a")


def test_cache_hit_and_miss(tmp_path) -> None:
    """Записанное чтение читается обратно, отсутствующий ключ даёт None."""
    cache_dir = tmp_path / "cache"
    key = "abc123"
    assert get_cached_reading(key, cache_dir=cache_dir) is None
    assert put_cached_reading(key, {"raw_text": "winery: X"}, cache_dir=cache_dir) is True
    cached = get_cached_reading(key, cache_dir=cache_dir)
    assert cached is not None and cached["raw_text"] == "winery: X" and cached["from_cache"] is True


def test_invalidation_by_model_and_prompt(tmp_path) -> None:
    """Смена модели или версии промпта даёт другой ключ — кэш не переиспользуется ошибочно."""
    photo = _photo(tmp_path)
    assert image_key(photo, "m1") != image_key(photo, "m2")
    assert image_key(photo, "m1", prompt_version="v1") != image_key(photo, "m1", prompt_version="v2")


def test_corrupt_cache_is_miss(tmp_path) -> None:
    """Повреждённый файл кэша трактуется как кэш-мисс, не роняя вызывающего."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "bad.json").write_text("{not valid json", encoding="utf-8")
    assert get_cached_reading("bad", cache_dir=cache_dir) is None


def test_clear_reader_cache(tmp_path) -> None:
    """Очистка удаляет все записи кэша и возвращает их число."""
    cache_dir = tmp_path / "cache"
    put_cached_reading("k1", {"raw_text": "a"}, cache_dir=cache_dir)
    put_cached_reading("k2", {"raw_text": "b"}, cache_dir=cache_dir)
    assert clear_reader_cache(cache_dir=cache_dir) == 2
    assert get_cached_reading("k1", cache_dir=cache_dir) is None


def test_observe_reading_uses_cache(tmp_path, monkeypatch) -> None:
    """Повторное чтение неизменного фото берётся из кэша: модель не вызывается второй раз."""
    photo = _photo(tmp_path)
    cache_dir = tmp_path / "cache"
    calls: list[int] = [0]

    def fake_observe(*args, **kwargs):  # noqa: ANN002, ANN003
        calls[0] += 1
        return {"ok": True, "raw_text": "winery: Тест", "latency_ms": 5.0, "reason": ""}

    monkeypatch.setattr("scripts.eval.llm_match_eval.observe_image", fake_observe)
    first = observe_reading(photo, "model-a", cache_dir=cache_dir)
    second = observe_reading(photo, "model-a", cache_dir=cache_dir)
    assert calls[0] == 1                      # модель вызвана один раз
    assert second["raw_text"] == "winery: Тест"
    assert second.get("from_cache") is True
    assert first.get("from_cache") is None    # первое чтение было живым
