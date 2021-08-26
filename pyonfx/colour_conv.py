import colorsys
import math
from typing import Final, Tuple

import numpy as np

from .types import Nb, Tup3


# Reference white
D65_XYZ_TRISTIMULUS_2: Final[Tup3[float]] = (0.95047, 1.00000, 1.08883)
D65_XYZ_TRISTIMULUS_10: Final[Tup3[float]] = (0.9481, 1.000, 1.07304)
κ: Final[float] = 24389 / 27
ϵ: Final[float] = 216 / 24389


def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    return colorsys.hls_to_rgb(h, l, s)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    return colorsys.hsv_to_rgb(h, s, v)


# -------------------------------------------------------------------------
# ---------------------------- RGB Conversions ----------------------------
# -------------------------------------------------------------------------
def rgb_to_xyz(r: float, g: float, b: float) -> Tuple[float, float, float]:
    # http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    rgb_mat = np.array((r, g, b), np.float64)
    conv_mat = np.array(
        [(0.4124564, 0.3575761, 0.1804375),
         (0.2126729, 0.7151522, 0.0721750),
         (0.0193339, 0.1191920, 0.9503041)],
        np.float64
    )
    inv_srgb_comp = np.where(
        rgb_mat <= 0.04045,
        rgb_mat / 12.92,
        ((rgb_mat + 0.055) / 1.055) ** 2.4
    )
    return tuple(np.dot(conv_mat, np.array([*inv_srgb_comp])))  # type: ignore


def rgb_to_xyy(r: float, g: float, b: float) -> Tuple[float, float, float]:
    xyz = rgb_to_xyz(r, g, b)
    return xyz_to_xyy(*xyz)


def rgb_to_lab(r: float, g: float, b: float) -> Tuple[float, float, float]:
    xyz = rgb_to_xyz(r, g, b)
    return xyz_to_lab(*xyz)


def rgb_to_lch_ab(r: float, g: float, b: float) -> Tuple[float, float, float]:
    xyz = rgb_to_xyz(r, g, b)
    return xyz_to_lch_ab(*xyz)


def rgb_to_luv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    xyz = rgb_to_xyz(r, g, b)
    return xyz_to_luv(*xyz)


def rgb_to_lch_uv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    xyz = rgb_to_xyz(r, g, b)
    return xyz_to_lch_uv(*xyz)


def rgb_to_hsl(r: float, g: float, b: float) -> Tuple[float, float, float]:
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, s, l


def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    return colorsys.rgb_to_hsv(r, g, b)


# -------------------------------------------------------------------------
# ---------------------------- XYZ Conversions ----------------------------
# -------------------------------------------------------------------------
def xyz_to_xyy(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_xyY.html
    if not x == y == z == 0.0:
        Y = y
        x, y = map(lambda a: a / (x + y + z), (x, y))
    else:
        Y = 0.0
        x, y = map(lambda a: a / sum(D65_XYZ_TRISTIMULUS_10), (x, y))
    return x, y, Y


def xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
    xr, yr, zr = [a / b for a, b in zip((x, y, z), D65_XYZ_TRISTIMULUS_10)]
    fx, fy, fz = [a ** (1/3) if a > ϵ else (κ * a + 16) / 116 for a in (xr, yr, zr)]
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def xyz_to_lch_ab(x: float, y: float, z: float) -> Tuple[float, float, float]:
    lab = xyz_to_lab(x, y, z)
    return lab_to_lch_ab(*lab)


def xyz_to_luv(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Luv.html
    x_ref, y_ref, z_ref = D65_XYZ_TRISTIMULUS_10
    yr = y / y_ref
    up = 4 * x / (x + 15 * y + 3 * z)
    vp = 9 * y / (x + 15 * y + 3 * z)
    up_ref = 4 * x_ref / (x_ref + 15 * y_ref + 3 * z_ref)
    vp_ref = 9 * y_ref / (x_ref + 15 * y_ref + 3 * z_ref)

    l = 116 * yr ** (1 / 3) - 16 if yr > ϵ else κ * yr
    u = 13 * l * (up - up_ref)
    v = 13 * l * (vp - vp_ref)

    return l, u, v


def xyz_to_lch_uv(x: float, y: float, z: float) -> Tuple[float, float, float]:
    luv = xyz_to_luv(x, y, z)
    return luv_to_lch_uv(*luv)


def xyz_to_rgb(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_RGB.html
    xyz_mat = np.array((x, y, z), np.float64)
    conv_mat = np.array(
        [(3.2404542, -1.5371385, -0.4985314),
         (-0.9692660, 1.8760108, 0.0415560),
         (0.0556434, -0.2040259, 1.0572252)],
        np.float64
    )
    linear_rgb = np.dot(conv_mat, xyz_mat)
    srgb_comp = np.where(
        linear_rgb <= 0.0031308,
        linear_rgb * 12.92,
        1.055 * linear_rgb ** (1 / 2.4) - 0.055
    )
    return tuple(map(lambda a: clamp_value(a, 0.0, 1.0), srgb_comp))  # type: ignore


# -------------------------------------------------------------------------
# ---------------------------- xyY Conversions ----------------------------
# -------------------------------------------------------------------------
def xyy_to_xyz(x: float, y: float, Y: float) -> Tuple[float, float, float]:
    # http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
    if Y == 0.0:
        x = y = z = 0.0
    else:
        z = ((1 - x - y) * Y) / y
        x = (x * Y) / y
        y = Y
    return x, y, z


def xyy_to_lab(x: float, y: float, Y: float) -> Tuple[float, float, float]:
    xyz = xyy_to_xyz(x, y, Y)
    return xyz_to_lab(*xyz)


def xyy_to_lch_ab(x: float, y: float, Y: float) -> Tuple[float, float, float]:
    lab = xyy_to_lab(x, y, Y)
    return lab_to_lch_ab(*lab)


def xyy_to_luv(x: float, y: float, Y: float) -> Tuple[float, float, float]:
    xyz = xyy_to_xyz(x, y, Y)
    return xyz_to_luv(*xyz)


def xyy_to_lch_uv(x: float, y: float, Y: float) -> Tuple[float, float, float]:
    luv = xyy_to_luv(x, y, Y)
    return luv_to_lch_uv(*luv)


def xyy_to_rgb(x: float, y: float, Y: float) -> Tuple[float, float, float]:
    xyz = xyy_to_xyz(x, y, Y)
    return xyz_to_rgb(*xyz)


# -------------------------------------------------------------------------
# ---------------------------- Lab Conversions ----------------------------
# -------------------------------------------------------------------------
def lab_to_xyz(l: float, a: float, b: float) -> Tuple[float, float, float]:
    # http://www.brucelindbloom.com/index.html?Eqn_Lab_to_XYZ.html
    fy = (l + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    yr = fy ** 3 if l > ϵ * κ else l / κ
    xr, zr = map(lambda n: n ** 3 if n ** 3 > ϵ else (116 * n - 16) / κ, (fx, fz))
    return tuple(n * m for n, m in zip((xr, yr, zr), D65_XYZ_TRISTIMULUS_10))  # type: ignore


def lab_to_xyy(l: float, a: float, b: float) -> Tuple[float, float, float]:
    xyz = lab_to_xyz(l, a, b)
    return xyz_to_xyy(*xyz)


def lab_to_lch_ab(l: float, a: float, b: float) -> Tuple[float, float, float]:
    # http://www.brucelindbloom.com/index.html?Eqn_Lab_to_LCH.html
    return l, math.sqrt(a ** 2 + b ** 2), (math.atan2(b, a) * 180 / math.pi) % 360


def lab_to_luv(l: float, a: float, b: float) -> Tuple[float, float, float]:
    xyz = lab_to_xyz(l, a, b)
    return xyz_to_luv(*xyz)


def lab_to_lch_uv(l: float, a: float, b: float) -> Tuple[float, float, float]:
    luv = lab_to_luv(l, a, b)
    return luv_to_lch_uv(*luv)


def lab_to_rgb(l: float, a: float, b: float) -> Tuple[float, float, float]:
    xyz = lab_to_xyz(l, a, b)
    return xyz_to_rgb(*xyz)


# -------------------------------------------------------------------------
# --------------------------- LCHab Conversions ---------------------------
# -------------------------------------------------------------------------
def lch_ab_to_xyz(l: float, c: float, h: float) -> Tuple[float, float, float]:
    lab = lch_ab_to_lab(l, c, h)
    return lab_to_xyy(*lab)


def lch_ab_to_xyy(l: float, c: float, h: float) -> Tuple[float, float, float]:
    xyz = lch_ab_to_xyz(l, c, h)
    return xyz_to_xyy(*xyz)


def lch_ab_to_lab(l: float, c: float, h: float) -> Tuple[float, float, float]:
    # http://www.brucelindbloom.com/index.html?Eqn_LCH_to_Lab.html
    hr = h * math.pi/180
    return l, math.cos(hr) * c, math.sin(hr) * c


def lch_ab_to_luv(l: float, c: float, h: float) -> Tuple[float, float, float]:
    xyz = lch_ab_to_xyz(l, c, h)
    return xyz_to_luv(*xyz)


def lch_ab_to_lch_uv(l: float, c: float, h: float) -> Tuple[float, float, float]:
    xyz = lch_ab_to_xyz(l, c, h)
    return xyz_to_lch_uv(*xyz)


def lch_ab_to_rgb(l: float, c: float, h: float) -> Tuple[float, float, float]:
    xyz = lch_ab_to_xyz(l, c, h)
    return xyz_to_rgb(*xyz)


# -------------------------------------------------------------------------
# ---------------------------- Luv Conversions ----------------------------
# -------------------------------------------------------------------------
def luv_to_xyz(l: float, u: float, v: float) -> Tuple[float, float, float]:
    x_ref, y_ref, z_ref = D65_XYZ_TRISTIMULUS_10
    u0 = 4 * x_ref / (x_ref + 15 * y_ref + 3 * z_ref)
    v0 = 9 * y_ref / (x_ref + 15 * y_ref + 3 * z_ref)

    y = ((l + 16) / 116) ** 3 if l > κ * ϵ else l / κ

    a = ((52 * l) / (u + 13 * l * u0) - 1) * (1 / 3)
    b = (-5) * y
    c = - 1 / 3
    d = ((39 * l) / (v + 13 * l * v0) - 5) * y

    x = (d - b) / (a - c)
    z = x * a + b

    return x, y, z 


def luv_to_xyy(l: float, u: float, v: float) -> Tuple[float, float, float]:
    xyz = luv_to_xyz(l, u, v)
    return xyz_to_xyy(*xyz)


def luv_to_lab(l: float, u: float, v: float) -> Tuple[float, float, float]:
    xyz = luv_to_xyz(l, u, v)
    return xyz_to_lab(*xyz)


def luv_to_lch_ab(l: float, u: float, v: float) -> Tuple[float, float, float]:
    lab = luv_to_lab(l, u, v)
    return lab_to_lch_ab(*lab)


def luv_to_lch_uv(l: float, u: float, v: float) -> Tuple[float, float, float]:
    # http://www.brucelindbloom.com/index.html?Eqn_Luv_to_LCH.html
    return l, math.sqrt(u ** 2 + v ** 2), (math.atan2(v, u) * 180 / math.pi) % 360


def luv_to_rgb(l: float, u: float, v: float) -> Tuple[float, float, float]:
    xyz = luv_to_xyz(l, u, v)
    return xyz_to_rgb(*xyz)


# -------------------------------------------------------------------------
# --------------------------- LCHuv Conversions ---------------------------
# -------------------------------------------------------------------------
def lch_uv_to_xyz(l: float, c: float, h: float) -> Tuple[float, float, float]:
    luv = lch_uv_to_luv(l, c, h)
    return luv_to_xyz(*luv)


def lch_uv_to_xyy(l: float, c: float, h: float) -> Tuple[float, float, float]:
    xyz = lch_uv_to_xyz(l, c, h)
    return xyz_to_xyy(*xyz)


def lch_uv_to_lab(l: float, c: float, h: float) -> Tuple[float, float, float]:
    xyz = lch_uv_to_xyz(l, c, h)
    return xyz_to_lab(*xyz)


def lch_uv_to_lch_ab(l: float, c: float, h: float) -> Tuple[float, float, float]:
    xyz = lch_uv_to_xyz(l, c, h)
    return xyz_to_lch_ab(*xyz)


def lch_uv_to_luv(l: float, c: float, h: float) -> Tuple[float, float, float]:
    # http://www.brucelindbloom.com/index.html?Eqn_LCH_to_Luv.html
    hr = h * math.pi/180
    return l, math.cos(hr) * c, math.sin(hr) * c


def lch_uv_to_rgb(l: float, c: float, h: float) -> Tuple[float, float, float]:
    xyz = lch_uv_to_xyz(l, c, h)
    return xyz_to_rgb(*xyz)


def clamp_value(val: Nb, min_val: Nb, max_val: Nb) -> Nb:
    """
    Clamp value val between min_val and max_val

    :param val:         Value to clamp
    :param min_val:     Minimum value
    :param max_val:     Maximum value
    :return:            Clamped value
    """
    return min(max_val, max(min_val, val))
