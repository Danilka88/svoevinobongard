# START_MODULE_CONTRACT
# PURPOSE: Варианты входа: неразрушающий full-frame набор для энкодера и multi-image набор для VLM
# SCOPE: build_variants, build_vlm_inputs
# DEPENDS: numpy, packages.common.preprocess, services.normalize.app.homography
# LINKS: M-NORMALIZE
# ROLE: RUNTIME
# MAP_MODE: EXPORTS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# build_variants - варианты для энкодера: полный кадр, letterbox 896 и 1280, кроп только за флагом
# build_vlm_inputs - набор для VLM: letterbox-база плюс pan-and-scan тайлы всего кадра
# END_MODULE_MAP
"""Варианты входа.

Правило проекта: **полный кадр — база**. Обрезка этикетки овальной формы, на
цилиндре или под углом теряет текст и блики, поэтому кроп-варп включается только
флагом ``NORMALIZE_CROP=1``. Вместо обрезки кадр масштабируется без искажения
(letterbox) и нарезается на перекрывающиеся тайлы по всей площади.

Реализация живёт в ``packages.common.preprocess`` (C-069): общий код нужен и VLM, а
сервис не должен импортировать другой сервис. Здесь остаётся тонкая обёртка,
подставляющая ``warp_quad`` для перспективного выпрямления кропа.
"""
from __future__ import annotations

import numpy as np

from packages.common.preprocess import (
    PreprocessConfig,
    build_variants as _build_variants,
    build_vlm_inputs,
)
from services.normalize.app.homography import warp_quad

__all__ = ["build_variants", "build_vlm_inputs"]


def build_variants(image: np.ndarray, quad: np.ndarray | None = None,
                   config: PreprocessConfig | None = None,
                   tagged: bool = False) -> list:
    """Собирает варианты входа для энкодера (см. packages.common.preprocess).

    Обёртка подставляет ``warp_quad`` сервиса нормализации, поэтому кроп за флагом
    ``PreprocessConfig.crop`` выпрямляет четырехугольник этикетки как раньше.
    """
    return _build_variants(image, quad, config=config, tagged=tagged, warper=warp_quad)
