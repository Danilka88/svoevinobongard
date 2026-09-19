# START_MODULE_CONTRACT
# PURPOSE: Нормализация текста для сопоставления: лат/кир транслит, OCR-двойники, алиасы сортов, типов и брендов
# SCOPE: RU_TO_LAT, LAT_TO_RU, LAT_ORDER, ALIASES, LOOKALIKES, fold, delookalike, cyr_to_lat,
#        lat_to_cyr, apply_aliases, forms, tokens, similarity
# DEPENDS: re, unicodedata, rapidfuzz
# LINKS: M-COMMON, M-MATCHER, M-OCR, M-TEXT, M-EVAL
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# RU_TO_LAT - посимвольная таблица русский -> латиница
# LAT_TO_RU - обратная таблица, многосимвольные диграфы
# LAT_ORDER - порядок применения обратной таблицы (длинные диграфы первыми)
# ALIASES - курируемые двусторонние соответствия: сорта, типы, цвета, регионы, термины, бренды
# LOOKALIKES - OCR-двойники латиница -> кириллица
# fold - базовая нормализация: NFKC, нижний регистр, диакритика, пунктуация, пробелы
# delookalike - замена OCR-двойников латиницы на кириллицу
# cyr_to_lat - перевод кириллицы в латиницу (для сопоставления со slug)
# lat_to_cyr - перевод латиницы в кириллицу (для сопоставления с названиями)
# apply_aliases - подстановка курируемых алиасов (сорта, типы, бренды)
# forms - набор канонических вариантов строки для сопоставления
# tokens - значимые токены с сохранением чисел и различающих слов
# similarity - максимум fuzzy-похожести по вариантам написания
# END_MODULE_MAP
"""Нормализация текста для сопоставления.

Каталог смешанный: винодельни и названия встречаются и латиницей (`WINEPARK`,
`Chateau Tamagne`), и кириллицей («Фанагория», «Шато Тамань»), а slug — почти всегда
латинская транслитерация русского написания (`pino-nuar`, `kaberne-sovinon`,
`shardone`). Поэтому простое приведение к нижнему регистру не находит совпадений,
и текстовый сигнал теряется.

Модуль даёт варианты написания одной и той же строки и сравнивает пары по максимуму
похожести. Это нормализация **только для сопоставления**: различающие токены (год,
«Рейнский», «Кюве», «Миллезимато») сохраняются, а отождествление по одному бренду
без названия или сорта попаданием не считается.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

RU_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e",
    "ю": "yu", "я": "ya",
}

LAT_TO_RU = {
    "sch": "щ", "shch": "щ", "zh": "ж", "kh": "х", "ch": "ч", "sh": "ш", "ts": "ц",
    "yu": "ю", "ya": "я", "yo": "ё", "ye": "е", "yi": "й", "ju": "ю", "ja": "я",
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "й", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "ы", "z": "з",
}
LAT_ORDER = tuple(sorted(LAT_TO_RU, key=len, reverse=True))

LOOKALIKES = {
    "c": "с", "o": "о", "p": "р", "a": "а", "e": "е", "h": "н", "y": "у", "b": "в",
    "m": "м", "t": "т", "k": "к", "n": "п", "x": "х", "3": "з", "0": "о", "1": "и",
}

# Курируемые соответствия: каноническая основа -> варианты написания (обе раскладки).
# Список покрывает то, что реально встречается в каталоге и на этикетках.
ALIASES: dict[str, tuple[str, ...]] = {
    # сорта
    "совиньон блан": ("sauvignon blanc", "sauvignon", "совиньон", "совиньон блан"),
    "пино нуар": ("pinot noir", "pino nuar", "пино нуар"),
    "пино гри": ("pinot grigio", "pino gri", "пино гри", "пино гриджио"),
    "шардоне": ("chardonnay", "shardone", "шардоне"),
    "рислинг": ("riesling", "risling", "рислинг", "risling reynskiy", "рислинг рейнский"),
    "саперави": ("saperavi", "саперави"),
    "ркацители": ("rkatsiteli", "ркацители"),
    "мускат": ("muscat", "muskat", "мускат"),
    "мерло": ("merlot", "merlo", "мерло"),
    "алиготе": ("aligote", "алиготе"),
    "кокур": ("kokur", "кокур"),
    "каберне совиньон": ("cabernet sauvignon", "kaberne sovinon", "каберне совиньон", "каберне"),
    "красностоп": ("krasnostop", "красностоп"),
    "сира": ("syrah", "sira", "сира", "шираз", "shiraz"),
    "глера": ("glera", "глера"),
    "бастардо": ("bastardo", "бастардо"),
    "мюллер тургау": ("muller thurgau", "muller-turgau", "мюллер тургау"),
    # типы и сладость
    "сухое": ("dry", "sukhoe", "сухое", "suhoe"),
    "полусухое": ("semi-dry", "polusukhoe", "полусухое"),
    "полусладкое": ("semi-sweet", "semisweet", "polusladkoe", "полусладкое"),
    "сладкое": ("sweet", "sladkoe", "сладкое"),
    "брют": ("brut", "bryut", "брют"),
    "экстра брют": ("extra brut", "ekstra bryut", "экстра брют"),
    "брют натюр": ("brut nature", "брют натюр"),
    # цвета
    "белое": ("white", "beloe", "белое"),
    "красное": ("red", "krasnoe", "красное"),
    "розовое": ("rose", "rozovoe", "розовое"),
    "оранжевое": ("orange", "oranzhevoe", "оранжевое"),
    # регионы
    "крым": ("crimea", "krym", "крым"),
    "кубань": ("kuban", "кубань"),
    "дон": ("don", "дон"),
    "дагестан": ("dagestan", "дагестан"),
    "ставрополь": ("stavropol", "ставрополь"),
    # общие термины
    "резерв": ("reserve", "rezerv", "резерв"),
    "кюве": ("cuvee", "kyuve", "кюве"),
    "миллезимато": ("millesimato", "millezimato", "миллезимато"),
    "блан де блан": ("blanc de blancs", "blanc de blanc", "блан де блан"),
    "блан де нуар": ("blanc de noirs", "blanc de noir", "блан де нуар"),
    "игристое": ("sparkling", "igristoe", "игристое"),
    "выдержанное": ("reserve", "vyderzhannoe", "выдержанное"),
    "красная книга": ("krasnaya kniga", "красная книга"),
    # бренды
    "шато тамань": ("chateau tamagne", "chateau taman", "shato taman", "шато тамань"),
    "фанагория": ("fanagoria", "fanagoriya", "fanagory", "фонагория", "фанагория"),
    "абрау дюрсо": ("abrau durso", "abrau-durso", "абрау дюрсо"),
    "новый свет": ("novy svet", "novyy svet", "новый свет"),
    "высокий берег": ("vysokiy bereg", "высокий берег"),
    "золотая балка": ("zolotaya balka", "золотая балка"),
    "лефкадия": ("lefkadia", "лефкадия"),
    "ведерников": ("vedernikov", "ведерников", "ведерниковъ"),
    "коктебель": ("koktebel", "коктебель"),
    "амфитрион": ("amfitrion", "амфитрион"),
    "мысхако": ("myskhako", "мысхако"),
    "геленджик": ("gelendzhik", "геленджик"),
    "кубань вино": ("kuban vino", "kuban-vino", "кубань вино"),
}
__all__ = ["ALIASES", "LAT_ORDER", "LAT_TO_RU", "LOOKALIKES", "RU_TO_LAT", "apply_aliases",
           "cyr_to_lat", "delookalike", "fold", "forms", "lat_to_cyr", "similarity", "tokens"]


def fold(text: str | None) -> str:
    """Базовая нормализация: NFKC, нижний регистр, без диакритики и пунктуации."""
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    # Диакритику снимаем только у латиницы: у кириллицы NFD разбирает «й» на «и» + бревис,
    # и «Рейнский» превращалось в «реинскии».
    cleaned: list[str] = []
    for char in value:
        decomposed = unicodedata.normalize("NFD", char)
        base = "".join(part for part in decomposed if unicodedata.category(part) != "Mn")
        # снимаем диакритику только там, где основа латинская (é -> e),
        # кириллица остаётся нетронутой (й НЕ превращается в и)
        cleaned.append(base if base and all("a" <= part <= "z" for part in base) else char)
    value = "".join(cleaned)
    value = value.replace("ё", "е").replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^0-9a-zа-я\s]+", " ", value).split())


def delookalike(text: str | None) -> str:
    """Заменяет OCR-двойники латиницы на кириллицу (обратная замена «кириллица-в-латиницу»)."""
    return "".join(LOOKALIKES.get(char, char) for char in fold(text))


def cyr_to_lat(text: str | None) -> str:
    """Переводит кириллицу в латиницу, чтобы сравнивать с транслит-слагами каталога."""
    result: list[str] = []
    for char in fold(text):
        result.append(RU_TO_LAT.get(char, char))
    return " ".join("".join(result).split())


def lat_to_cyr(text: str | None) -> str:
    """Переводит латиницу в кириллицу, чтобы сравнивать с русскими названиями."""
    value = fold(text)
    result: list[str] = []
    index = 0
    while index < len(value):
        for size in (4, 3, 2):
            chunk = value[index:index + size]
            if chunk in LAT_TO_RU and len(chunk) == size:
                result.append(LAT_TO_RU[chunk])
                index += size
                break
        else:
            char = value[index]
            result.append(LAT_TO_RU.get(char, char))
            index += 1
    return " ".join("".join(result).split())


def apply_aliases(text: str | None) -> str:
    """Подставляет канонические основы вместо любых вариантов написания.

    Замена идёт по границам слов и от длинных вариантов к коротким, чтобы
    «совиньон блан» не превращался в «совиньон блан блан».
    """
    value = fold(text)
    pairs: list[tuple[str, str]] = []
    for canonical, variants in ALIASES.items():
        for variant in variants + (canonical,):
            pairs.append((fold(variant), canonical))
    seen: set[str] = set()
    for variant, canonical in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
        if not variant or variant in seen:
            continue
        seen.add(variant)
        value = re.sub(rf"(?<![0-9a-zа-я]){re.escape(variant)}(?![0-9a-zа-я])", canonical, value)
    return " ".join(value.split())


def forms(text: str | None) -> list[str]:
    """Набор канонических вариантов строки; матчинг берёт максимум похожести по ним."""
    base = fold(text)
    if not base:
        return []
    candidates = [
        base,
        delookalike(base),
        cyr_to_lat(base),
        lat_to_cyr(base),
        apply_aliases(base),
        apply_aliases(lat_to_cyr(base)),
        apply_aliases(cyr_to_lat(base)),
    ]
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def tokens(text: str | None) -> list[str]:
    """Значимые токены: числа и различающие слова сохраняются.

    Год, «рейнский», «кюве», «миллезимато» и подобные остаются отдельными токенами,
    поэтому нормализация не склеивает разные вина одной линейки.
    """
    return [token for token in fold(text).split() if len(token) >= 2]


def similarity(left: str | None, right: str | None) -> float:
    """Максимум похожести по вариантам написания (0..1).

    Сравниваются все пары вариантов левой и правой строк: это позволяет сопоставить
    «Пино Нуар» с `pino-nuar`, «Шато Тамань» с `chateau tamagne` и латинскую этикетку
    с русским названием каталога.
    """
    left_forms, right_forms = forms(left), forms(right)
    if not left_forms or not right_forms:
        return 0.0
    best = 0.0
    for left_value in left_forms[:5]:
        for right_value in right_forms[:5]:
            best = max(best, fuzz.token_set_ratio(left_value, right_value) / 100.0)
    return round(best, 4)
