# START_MODULE_CONTRACT
# PURPOSE: Стража governance: ни один исходный файл проекта не должен остаться без семантической разметки
# SCOPE: ROOTS, SKIP_PARTS, ALLOWED_WITHOUT_MARKUP, MARKER, iter_python_files, missing_markup
# DEPENDS: pathlib
# LINKS: M-DATA
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ROOTS - каталоги исходного кода проекта
# SKIP_PARTS - части пути, которые не проверяются (vendored и служебные)
# ALLOWED_WITHOUT_MARKUP - файлы, которым разметка не нужна
# MARKER - обязательный маркер семантической разметки
# iter_python_files - перебирает исходные файлы проекта
# missing_markup - возвращает файлы без разметки
# test_all_source_files_have_markup - Разметка обязательна: иначе файл невидим проверке целостности
# test_guard_covers_real_sources - Стража осмысленна: она действительно видит исходники, а не пустой список
# test_vendored_models_are_excluded - Vendored модели исключены: их разметка не наша ответственность
# END_MODULE_MAP
"""Стража governance: слепой участок проверки становится видимым в pytest.

GRACE-проверка видит только файлы, у которых уже есть семантическая разметка: файл без разметки
невидим гейту. Этот тест перебирает исходники проекта и падает, если у файла разметки нет, поэтому
забыть её больше нельзя, независимо от внешнего CLI.
"""


from __future__ import annotations

from pathlib import Path

__all__ = [
    "ROOTS",
    "SKIP_PARTS",
    "ALLOWED_WITHOUT_MARKUP",
    "MARKER",
    "iter_python_files",
    "missing_markup",
    "test_all_source_files_have_markup",
    "test_guard_covers_real_sources",
    "test_vendored_models_are_excluded",
]

ROOTS = ("packages", "services", "scripts", "tests")
SKIP_PARTS = ("data/models", "node_modules", ".venv", "__pycache__", ".git")
ALLOWED_WITHOUT_MARKUP = {"__init__.py"}
MARKER = "START_MODULE_CONTRACT"

_ROOT = Path(__file__).resolve().parents[1]


def iter_python_files() -> list[Path]:
    """Перебирает исходные файлы проекта, исключая vendored и служебные каталоги."""
    files: list[Path] = []
    for root in ROOTS:
        for path in sorted((_ROOT / root).rglob("*.py")):
            relative = path.relative_to(_ROOT).as_posix()
            if any(part in relative for part in SKIP_PARTS):
                continue
            if path.name in ALLOWED_WITHOUT_MARKUP:
                continue
            body = path.read_text(encoding="utf-8")
            if not body.strip() or all(line.strip().startswith("#") or not line.strip()
                                       for line in body.splitlines()):
                continue  # пустышка без кода
            files.append(path)
    return files


def missing_markup() -> list[str]:
    """Возвращает файлы проекта, у которых нет семантической разметки."""
    return [path.relative_to(_ROOT).as_posix() for path in iter_python_files()
            if MARKER not in path.read_text(encoding="utf-8")]


def test_all_source_files_have_markup() -> None:
    """Разметка обязательна: иначе файл невидим проверке целостности."""
    missing = missing_markup()
    assert missing == [], "файлы без семантической разметки: " + ", ".join(missing)


def test_guard_covers_real_sources() -> None:
    """Стража осмысленна: она действительно видит исходники, а не пустой список."""
    files = iter_python_files()
    assert len(files) > 100, f"проверено файлов: {len(files)}"
    assert any(path.name == "main.py" for path in files)
    assert any(str(path).endswith("tests/test_governance_markup.py") for path in files)


def test_vendored_models_are_excluded() -> None:
    """Vendored модели исключены: их разметка не наша ответственность."""
    files = [path.relative_to(_ROOT).as_posix() for path in iter_python_files()]
    assert not [name for name in files if "data/models" in name]
