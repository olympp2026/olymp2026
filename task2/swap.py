import argparse
from dataclasses import dataclass

import cv2
import numpy as np


# ============================================================
#  Константы и структуры данных
# ============================================================

@dataclass
class MaskParams:
    """
    Параметры пороговой маски в HSV.

    в OpenCV Hue лежит в диапазоне [0..179], Saturation/Value — [0..255].
    """
    h1: int
    h2: int
    s_min: int = 40
    v_min: int = 40


# ============================================================
#  Вспомогательные функции: ввод/вывод и безопасные преобразования
# ============================================================

def read_bgr(path: str) -> np.ndarray:
    """
    Считывает изображение с диска в формате BGR
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return img


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """
    Обрезает значения в [0..255] и приводит к uint8 (для сохранения изображений).
    """
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
#  Построение бинарных масок (HSV) и постобработка
# ============================================================

def hsv_mask(img_bgr: np.ndarray, params: MaskParams) -> np.ndarray:
    """
    Строит бинарную маску {0,255} по диапазону HSV.

    Поддерживается "оборачивающийся" диапазон по Hue (например, 170..179 и 0..10).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    sv_ok = (s >= params.s_min) & (v >= params.v_min)

    if params.h1 <= params.h2:
        h_ok = (h >= params.h1) & (h <= params.h2)
    else:
        # Оборачивание через 0 (wrap-around)
        h_ok = (h >= params.h1) | (h <= params.h2)

    return ((h_ok & sv_ok).astype(np.uint8) * 255)


def remove_sky_like(mask: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
    """
    Убирает из маски "похожее на небо/облака":
    яркие (V высокий) и области с низким S (с тусклым цветом).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    sky_like = (s < 35) & (v > 170)

    out = mask.copy()
    out[sky_like] = 0
    return out


def postprocess_mask(mask: np.ndarray, k_close: int = 9, k_open: int = 5) -> np.ndarray:
    """
    Морфологическая очистка маски:
    - CLOSE: заделывает дырки/разрывы
    - OPEN: удаляет мелкий шум
    """
    m = mask.copy()

    if k_close > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=1)

    if k_open > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, iterations=1)

    return m


def feather_mask(mask: np.ndarray, blur: int = 17) -> np.ndarray:
    """
    Добавляет альфа канал, делает маску частично "прозрачной"
    - вход: {0,255}
    - выход: float в [0..1]
    """
    if blur % 2 == 0:
        blur += 1

    m = cv2.GaussianBlur(mask.astype(np.float32), (blur, blur), 0)
    return np.clip(m / 255.0, 0.0, 1.0)


# ============================================================
#  Цветовые преобразования
# ============================================================

def reinhard_transfer_lab_ab_only(
    src_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    src_mask: np.ndarray,
    ref_mask: np.ndarray,
    *,
    strength: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Перенос цветовой статистики Reinhard в пространстве Lab,
    но только по каналам a,b (цветность), без изменения яркости.


    strength:
      0   -> без изменений
      1.0 -> полный перенос статистики
    """
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    sm = (src_mask > 0)
    rm = (ref_mask > 0)

    if sm.sum() < 1000:
        raise ValueError("Слишком маленькая маска источника (src_mask)")
    if rm.sum() < 1000:
        raise ValueError("Слишком маленькая маска референса (ref_mask)")

    out = src_lab.copy()

    # Только a и b, индексы 1 и 2
    for c in (1, 2):
        s_vals = src_lab[..., c][sm]
        r_vals = ref_lab[..., c][rm]

        s_mean, s_std = float(np.mean(s_vals)), float(np.std(s_vals))
        r_mean, r_std = float(np.mean(r_vals)), float(np.std(r_vals))

        s_std = max(s_std, eps)
        r_std = max(r_std, eps)

        transformed = (src_lab[..., c] - s_mean) / s_std * r_std + r_mean

        out[..., c] = np.where(
            sm,
            (1.0 - strength) * src_lab[..., c] + strength * transformed,
            src_lab[..., c],
        )

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def hsv_autumn_recolor(
    src_bgr: np.ndarray,
    leaf_mask: np.ndarray,
    hue_target: int,
    sat_boost: float,
    val_scale: float,
) -> np.ndarray:
    """
    Прямое перекрашивание листвы в "осенний" цвет в HSV.

    Алгоритм:
    - Hue ставим в заданный оттенок (примерно 15..25 — оранжевый),
    - Saturation усиливаем
    """
    hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    m = (leaf_mask > 0)

    # Hue: принудительно задаём осенний оттенок
    h[m] = float(hue_target)

    # S/V: усиливаем насыщенность и затемняем
    s[m] = np.clip(s[m] * sat_boost, 0, 255)
    v[m] = np.clip(v[m] * val_scale, 0, 255)

    hsv2 = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)


# ============================================================
#  Маски и альфа для двух направлений преобразования
# ============================================================

def build_leaf_alpha_for_summer_to_autumn(summer_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит маску листвы (зелёной) на летнем фото и соответствующую альфу для смешивания.

    Алгоритм:
    - базовая зеленая маска,
    - дополнительное расширение вниз/в тень (dilate + фильтр extra_ok),
    - усиление альфы внутри листвы, чтобы hue_target был заметен,
    - обнуление альфы на стволах (пиксели с низким S/V).
    """
    hsv = cv2.cvtColor(summer_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    # 0) Базовая зелёная маска
    green0 = hsv_mask(summer_bgr, MaskParams(h1=35, h2=95, s_min=60, v_min=55))
    green0 = remove_sky_like(green0, summer_bgr)
    green0 = postprocess_mask(green0, k_close=9, k_open=7)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)) # параметр подобран вручную
    green_core = cv2.erode(green0, ker, iterations=1)

    # 2) расширяем маску, чтобы захватить нижнюю/теневую листву
    ker_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    green_extra = cv2.dilate(green0, ker_d, iterations=1)

    # Оставляем в extra только "похожие на листья" пиксели
    extra_ok = (s >= 80) & (v >= 45)  # параметры подобраны вручную, чтобы захватить нижние листья
    green_extra = np.where(extra_ok, green_extra, 0).astype(np.uint8)

    # Итоговая маска листвы
    green = cv2.bitwise_or(green_core, green_extra)

    # 3) настройка альфа + усиление (чтобы цвет был заметнее)
    alpha = feather_mask(green, blur=11)
    alpha = np.clip(alpha * 1.8, 0.0, 1.0)

    # 4) Защита стволов/веток: где низкие S или V -> альфа = 0
    trunk_like = (s < 80) | (v < 45)
    alpha[trunk_like] = 0.0

    return green, alpha


def build_autumn_mask_on_photo1(autumn_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит маску осенней листвы на Photo1 (осень),
    а также альфу для смешивания (осень -> лето).
    """
    warm1 = hsv_mask(autumn_bgr, MaskParams(h1=0, h2=35, s_min=45, v_min=35))
    warm2 = hsv_mask(autumn_bgr, MaskParams(h1=36, h2=50, s_min=35, v_min=35))
    warm = cv2.bitwise_or(warm1, warm2)

    warm = remove_sky_like(warm, autumn_bgr)
    warm = postprocess_mask(warm, k_close=13, k_open=7)

    alpha = feather_mask(warm, blur=21)
    alpha = np.clip(alpha * 1.3, 0.0, 1.0)

    return warm, alpha


# ============================================================
#  Смешивание результатов
# ============================================================

def alpha_blend(base_bgr: np.ndarray, changed_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
   Накладывает перекрашенный слой на исходник
    """
    a = alpha[..., None].astype(np.float32)
    out = base_bgr.astype(np.float32) * (1.0 - a) + changed_bgr.astype(np.float32) * a
    return ensure_uint8(out)


# ============================================================
#  CLI / Точка входа
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Season swap: Photo1(autumn)->Summer, Photo2(summer)->Autumn")
    parser.add_argument("--photo1", default="Photo1.jpg", help="Осеннее фото (по умолчанию: Photo1.jpg)")
    parser.add_argument("--photo2", default="Photo2.jpg", help="Летнее фото (по умолчанию: Photo2.jpg)")
    parser.add_argument("--out_summer", default="Summer.jpg", help="Выход: осень->лето (по умолчанию: Summer.jpg)")
    parser.add_argument("--out_autumn", default="Autumn.jpg", help="Выход: лето->осень (по умолчанию: Autumn.jpg)")
    parser.add_argument("--strength", type=float, default=1.2)
    parser.add_argument("--hue_target", type=int, default=18, help=" (оранжевый ~ 15..25)")
    parser.add_argument("--save_masks", action="store_true", help="Сохранить маски и альфа - маски для отладки")
    args = parser.parse_args()

    autumn = read_bgr(args.photo1)  # Photo1
    summer = read_bgr(args.photo2)  # Photo2

    # Маски/альфы
    autumn_mask, autumn_alpha = build_autumn_mask_on_photo1(autumn)
    summer_mask, summer_alpha = build_leaf_alpha_for_summer_to_autumn(summer)

    # Осень -> Лето
    autumn_changed = reinhard_transfer_lab_ab_only(
        src_bgr=autumn,
        ref_bgr=summer,
        src_mask=autumn_mask,
        ref_mask=summer_mask,
        strength=float(args.strength),
    )
    out_summer = alpha_blend(autumn, autumn_changed, autumn_alpha)
    cv2.imwrite(args.out_summer, out_summer)

    # Лето -> Осень (HSV перекраска листвы) ---
    summer_orange = hsv_autumn_recolor(
        src_bgr=summer,
        leaf_mask=summer_mask,
        hue_target=int(args.hue_target),
        sat_boost=1.65,
        val_scale=0.97,
    )
    out_autumn = alpha_blend(summer, summer_orange, summer_alpha)
    cv2.imwrite(args.out_autumn, out_autumn)

    # отладка
    if args.save_masks:
        cv2.imwrite("mask_autumn.png", autumn_mask)
        cv2.imwrite("mask_summer.png", summer_mask)
        cv2.imwrite("alpha_autumn.png", ensure_uint8(autumn_alpha * 255))
        cv2.imwrite("alpha_summer.png", ensure_uint8(summer_alpha * 255))

    print("Готово.")
    print(f"Сохранено: {args.out_summer}")
    print(f"Сохранено: {args.out_autumn}")


if __name__ == "__main__":
    main()
