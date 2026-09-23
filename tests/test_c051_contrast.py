"""Тесты C-051: контраст токенов интерфейса по WCAG AA.

Тест читает `apps/web/assets/css/main.css`, сам извлекает токены светлой и тёмной темы и пары
«цвет текста на фоне» прямо из правил CSS, поэтому будущие изменения токенов ловятся без правок
теста. Порог: 4.5 для текста, 3.0 для нетекстовых пар (границы и подобное). Значения не
хардкодятся — единственный источник истины это файл стилей.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

CSS_PATH = Path(__file__).resolve().parents[1] / "apps" / "web" / "assets" / "css" / "main.css"
TEXT_MIN = 4.5
UI_MIN = 3.0

# Пары, которые относятся к нетекстовым (границы, разделители): к ним применяется порог 3.0.
UI_SELECTOR_HINTS = ("border", "divider", "outline", "ring", "focus")


def _blocks(css: str) -> list[tuple[str, str]]:
    """Возвращает список (селектор, тело) для всех блоков верхнего уровня."""
    result: list[tuple[str, str]] = []
    depth = 0
    selector = ""
    body_start = 0
    index = 0
    while index < len(css):
        char = css[index]
        if char == "{" and depth == 0:
            selector = css[body_start:index].strip().split("\n")[-1].strip()
            depth = 1
            body_start = index + 1
        elif char == "{":
            depth += 1
        elif char == "}" and depth == 1:
            result.append((selector, css[body_start:index]))
            depth = 0
            body_start = index + 1
        elif char == "}":
            depth -= 1
        index += 1
    return result


def _tokens(body: str) -> dict[str, str]:
    """Извлекает токены вида --имя: значение из тела блока."""
    return {name: value.strip()
            for name, value in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", body, re.IGNORECASE)}


def _resolve(value: str, tokens: dict[str, str], depth: int = 0) -> str:
    """Разворачивает ссылки var(--token) до конкретного значения."""
    if depth > 8:
        return value
    match = re.fullmatch(r"var\((--[a-z0-9-]+)(?:\s*,\s*(.+))?\)", value.strip(), re.IGNORECASE)
    if not match:
        return value.strip()
    name, fallback = match.group(1), match.group(2)
    if name in tokens:
        return _resolve(tokens[name], tokens, depth + 1)
    return _resolve(fallback, tokens, depth + 1) if fallback else ""


def _hex(value: str) -> tuple[int, int, int] | None:
    """Преобразует #rgb/#rrggbb в компоненты; иные форматы не поддерживаются."""
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", text):
        return None
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _luminance(rgb: tuple[int, int, int]) -> float:
    def channel(value: int) -> float:
        share = value / 255
        return share / 12.92 if share <= 0.03928 else ((share + 0.055) / 1.055) ** 2.4
    red, green, blue = (channel(part) for part in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(foreground: str, background: str) -> float:
    """Контраст по WCAG 2.1 для двух цветов в шестнадцатеричной записи."""
    first, second = _hex(foreground), _hex(background)
    if first is None or second is None:
        return 0.0
    light, dark = sorted((_luminance(first), _luminance(second)), reverse=True)
    return round((light + 0.05) / (dark + 0.05), 2)


def _themes() -> dict[str, dict[str, str]]:
    """Собирает токены светлой и тёмной темы: тёмная наследует незаданные значения."""
    css = CSS_PATH.read_text(encoding="utf-8")
    light: dict[str, str] = {}
    dark: dict[str, str] = {}
    for selector, body in _blocks(css):
        found = _tokens(body)
        if not found:
            continue
        if selector.startswith(":root"):
            light.update(found)
        elif "dark" in selector.lower():
            dark.update(found)
    merged_dark = {**light, **dark}
    return {"light": light, "dark": merged_dark}


def _pairs() -> list[tuple[str, str, str]]:
    """Пары «текст на фоне» из правил CSS: селектор, токен текста, токен фона."""
    css = CSS_PATH.read_text(encoding="utf-8")
    pairs: list[tuple[str, str, str]] = []
    for selector, body in _blocks(css):
        color = re.search(r"(?<!-)color\s*:\s*(var\(--[a-z0-9-]+\))", body, re.IGNORECASE)
        background = re.search(r"background(?:-color)?\s*:\s*(var\(--[a-z0-9-]+\))", body, re.IGNORECASE)
        if color and background:
            pairs.append((selector, color.group(1), background.group(1)))
    return pairs


def _report() -> list[tuple[str, str, str, str, float, float]]:
    """Считает контраст всех найденных пар в обеих темах."""
    themes = _themes()
    rows: list[tuple[str, str, str, str, float, float]] = []
    for selector, fg, bg in _pairs():
        for theme in ("light", "dark"):
            tokens = themes[theme]
            foreground = _resolve(fg, tokens)
            background = _resolve(bg, tokens)
            ratio = contrast(foreground, background)
            rows.append((selector, theme, foreground, background, ratio,
                         UI_MIN if any(hint in selector.lower() for hint in UI_SELECTOR_HINTS) else TEXT_MIN))
    return rows


def test_tokens_are_parsed() -> None:
    """Проверка, что тест действительно читает файл стилей, а не пустой набор."""
    themes = _themes()
    assert len(themes["light"]) > 20, "токены светлой темы не найдены"
    assert len(themes["dark"]) > 20, "токены тёмной темы не найдены"
    assert _pairs(), "пары «текст на фоне» не найдены"


def test_contrast_report_is_printed() -> None:
    """Печатает таблицу пар с числами: это отчёт для ревью, он же доказательство."""
    rows = _report()
    print("\nселектор | тема | текст | фон | контраст | порог")
    for selector, theme, foreground, background, ratio, minimum in rows:
        mark = "OK " if ratio >= minimum else "FAIL"
        print(f"{mark} {selector[:42]:<44} {theme:<5} {foreground} на {background} = {ratio} (≥{minimum})")
    assert rows


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_all_token_pairs_meet_wcag_aa(theme: str) -> None:
    """Каждая пара «текст на фоне» обязана проходить порог: регрессия контраста недопустима."""
    failures = [(selector, foreground, background, ratio, minimum)
                for selector, row_theme, foreground, background, ratio, minimum in _report()
                if row_theme == theme and ratio < minimum]
    assert not failures, "\n".join(
        f"{selector}: {foreground} на {background} = {ratio} < {minimum}"
        for selector, foreground, background, ratio, minimum in failures)


def test_badges_are_not_removed() -> None:
    """Запрещено «чинить» контраст удалением бейджей: они обязаны остаться в стилях."""
    css = CSS_PATH.read_text(encoding="utf-8")
    for selector in (".badge-gold", ".badge-warning", ".badge-success", ".badge-error",
                     ".nav-link.active", ".chip-accent"):
        assert selector in css, f"{selector} исчез из стилей"


def test_soft_background_text_uses_dedicated_token() -> None:
    """Текст на мягком фоне берёт отдельный токен, а не светлый акцент."""
    css = CSS_PATH.read_text(encoding="utf-8")
    for token in ("--accent-on-soft", "--gold-on-soft", "--success-on-soft",
                  "--warning-on-soft", "--error-on-soft"):
        assert token in css, f"нет токена {token}"
    for selector in (".badge-gold", ".badge-warning", ".badge-success", ".badge-error",
                     ".chip-accent", ".nav-link.active"):
        rule = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
        assert rule, f"правило {selector} не найдено"
        assert "-on-soft" in rule.group(1), f"{selector}: текст на мягком фоне должен брать *-on-soft"
