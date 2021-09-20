"""Colourspace module"""
from __future__ import annotations

__all__ = [
    'RGBS', 'RGBAS',
    'RGB', 'RGB24', 'RGB30', 'RGB36', 'RGB42', 'RGB48',
    'RGBA', 'RGBA32', 'RGBA40', 'RGBA48', 'RGBA56', 'RGBA64',
    'HSL', 'HSV',
    'HTML', 'ASSColor',
    'XYZ', 'xyY', 'Lab', 'LCHab', 'Luv', 'LCHuv'
]

import re
from abc import ABC, abstractmethod
from typing import (Any, ClassVar, Dict, List, Sequence, Tuple, Type, TypeVar,
                    cast, overload)

from .convert import ConvertColour as CC
from .misc import clamp_value
from .types import (ACV, Nb, Nb8bit, Pct, TCV_co, Tup3, Tup3Str, Tup4,
                    check_annotations)

TRGB = TypeVar('TRGB', bound='BaseRGB[Any]')  # type: ignore
THSX = TypeVar('THSX', bound='HueSaturationBased')  # Type Hue Saturation ___
TCS = TypeVar('TCS', bound='ColourSpace[Any]')


class ColourSpace(Sequence[TCV_co], ABC):
    """Base class for colourspace interface"""

    _colour_values: Dict[str, TCV_co]

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not hasattr(self, '_colour_values'):
            self._colour_values = {}
        super().__init__()

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith('_'):
            self._colour_values[name] = value
        super().__setattr__(name, value)

    @overload
    def __getitem__(self, x: int) -> TCV_co:
        ...

    @overload
    def __getitem__(self, x: slice) -> Tuple[TCV_co, ...]:
        ...

    def __getitem__(self, x: int | slice) -> TCV_co | Tuple[TCV_co, ...]:
        return tuple(self._colour_values.values())[x]

    def __len__(self) -> int:
        return len(self._colour_values.values())

    def __eq__(self, o: object) -> bool:
        response = False
        if isinstance(o, ColourSpace):
            response = all(
                a == b
                for a, b in zip(
                    self._colour_values.items(), o._colour_values.items()
                )
            )
        return response

    def __repr__(self) -> str:
        return str(tuple(self._colour_values.values()))

    def __str__(self) -> str:
        return '\n'.join(map(str, self._colour_values.items()))

    @abstractmethod
    def interpolate(self, nobj: TCS, pct: Pct, /) -> TCS:
        """
        Interpolate the colour values of the current object with nobj
        and returns a new interpolated object.

        :param nobj:            Second colourspace. Must be of the same type
        :param pct:             Percentage value in the range 0.0 - 1.0
        :return:                New colourspace object
        """
        ...

    @abstractmethod
    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        """
        Convert current object to an RGB type object

        :param rgb_type:    RGB type
        :return:            RGB object
        """
        ...

    @abstractmethod
    def to_xyz(self) -> XYZ:
        """
        Convert current object to a XYZ object

        :return:            XYZ object
        """
        ...

    @abstractmethod
    def to_xyy(self) -> xyY:
        """
        Convert current object to a xyY object

        :return:            xyY object
        """
        ...

    @abstractmethod
    def to_lab(self) -> Lab:
        """
        Convert current object to a Lab object

        :return:            Lab object
        """
        ...

    @abstractmethod
    def to_lch_ab(self) -> LCHab:
        """
        Convert current object to a LCHab object

        :return:            LCHab object
        """
        ...

    @abstractmethod
    def to_luv(self) -> Luv:
        """
        Convert current object to a Luv object

        :return:            Luv object
        """
        ...

    @abstractmethod
    def to_lch_uv(self) -> LCHuv:
        """
        Convert current object to a LCHuv object

        :return:            LCHuv object
        """
        ...

    @abstractmethod
    def to_hsl(self) -> HSL:
        """
        Convert current object to a HSL object

        :return:            HSL object
        """
        ...

    @abstractmethod
    def to_hsv(self) -> HSV:
        """
        Convert current object to a HSV object

        :return:            HSV object
        """
        ...

    @abstractmethod
    def to_html(self) -> HTML:
        """
        Convert current object to a HTML object

        :return:            HTML object
        """
        ...

    @abstractmethod
    def to_ass_color(self) -> ASSColor:
        """
        Convert current object to an ASSColor object

        :return:            AssColor object
        """
        ...


class NumBased(ColourSpace[Nb], ABC):
    """Number based colourspace"""

    def interpolate(self, nobj: TCS, pct: Pct, /) -> TCS:
        vals: List[ACV] = []
        for (cs1_name, cs1_val), (cs2_name, cs2_val) in zip(self._colour_values.items(), nobj._colour_values.items()):
            if cs1_name == cs2_name:
                vals.append((1 - pct) * cs1_val + pct * cs2_val)
            else:
                raise ValueError(f'interpolate: attribute names must be identical! -> {cs1_name} != {cs2_name}')
        return cast(TCS, self.__class__(tuple(vals)))


class ForceNumber(NumBased[Nb], ABC):
    """Base class for clamping and forcing type values"""

    peaks: ClassVar[Tuple[Nb, Nb]]
    """Max value allowed"""

    force_type: ClassVar[Type[Nb]]
    """Forcing type"""

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith('_'):
            value = clamp_value(
                self.force_type(value),
                self.force_type(self.peaks[0]),
                self.force_type(self.peaks[1])
            )
        super().__setattr__(name, value)


class ForceFloat(ForceNumber[float], ABC):
    """Force values to float and clamp in the range peaks"""

    force_type = float

    def round(self, ndigits: int) -> None:
        """
        Round a number to a given precision in decimal digits.

        :param ndigits:         Number of digits
        """
        for name, val in self._colour_values.items():
            setattr(self, name, round(val, ndigits))


class ForceInt(ForceNumber[int], ABC):
    """Force values to int (truncate them if necessary) and clamp in the range peaks"""
    force_type = int


class BaseRGB(ColourSpace[Nb], ABC):
    """Base class for RGB colourspaces"""
    r: Nb
    """Red value"""
    g: Nb
    """Green value"""
    b: Nb
    """Blue value"""

    peaks: ClassVar[Tuple[Nb, Nb]]
    """Max value allowed"""

    def __new__(cls: Type[TRGB], _x: ColourSpace[TCV_co] | Tuple[Nb, ...]) -> TRGB:
        return _x.to_rgb(cls) if not isinstance(_x, tuple) else super().__new__(cls)

    def __init__(self, _x: ColourSpace[TCV_co] | Tuple[Nb, ...]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.r, self.g, self.b = _x

    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        if type(self) == rgb_type:
            return cast(TRGB, self)

        newpeaks = rgb_type.peaks

        nvalues = tuple((1 / self.peaks[1]) * val for val in self)
        svalues = tuple(v * newpeaks[1] for v in nvalues)

        if newpeaks[1] != 1:
            svalues = tuple(round(sval) for sval in svalues)

        return rgb_type(svalues)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.rgb_to_xyz(*self.to_rgb(RGBS)))

    def to_xyy(self) -> xyY:
        return xyY(CC.rgb_to_xyy(*self.to_rgb(RGBS)))

    def to_lab(self) -> Lab:
        return Lab(CC.rgb_to_lab(*self.to_rgb(RGBS)))

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.rgb_to_lch_ab(*self.to_rgb(RGBS)))

    def to_luv(self) -> Luv:
        return Luv(CC.rgb_to_luv(*self.to_rgb(RGBS)))

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.rgb_to_lch_uv(*self.to_rgb(RGBS)))

    def to_hsl(self) -> HSL:
        return HSL(CC.rgb_to_hsl(*self.to_rgb(RGBS)))

    def to_hsv(self) -> HSV:
        return HSV(CC.rgb_to_hsv(*self.to_rgb(RGBS)))

    def to_html(self) -> HTML:
        r, g, b = self.to_rgb(RGB)
        return HTML((r, g, b))

    def to_ass_color(self) -> ASSColor:
        return ASSColor(
            '&H'
            + ''.join(hex(x)[2:].zfill(2) for x in list(self.to_rgb(RGB))[::-1])
            + '&'
        )


class RGBNoAlpha(BaseRGB[Nb], ABC):
    """Base class for RGB colourspaces without alpha"""

    def __new__(cls, _x: ColourSpace[ACV] | Tup3[Nb]) -> RGBNoAlpha[Nb]:
        """
        Make a new RGB colourspace object

        :param _x:          Colourspace object or tuple of three numbers R, G and B values
        """
        return super().__new__(cls, _x)

    def __init__(self, _x: ColourSpace[ACV] | Tup3[Nb]) -> None:
        """
        Make a new RGB colourspace object

        :param _x:          Colourspace object or tuple of three numbers R, G and B values
        """
        super().__init__(_x)


class RGBAlpha(BaseRGB[Nb], ABC):
    """Base class for RGB colourspaces with alpha"""

    a: Nb
    """Alpha value"""

    def __new__(cls, _x: ColourSpace[ACV] | Tup3[Nb] | Tup4[Nb]) -> RGBAlpha[Nb]:
        """
        Make a new RGB colourspace object

        :param _x:          Colourspace object
                            or tuple of three numbers R, G and B values
                            or tuple of four numbers R, G, B and Alpha values
        """
        return super().__new__(cls, _x)

    def __init__(self, _x: ColourSpace[ACV] | Tup3[Nb] | Tup4[Nb]) -> None:
        """
        Make a new RGB colourspace object

        :param _x:          Colourspace object
                            or tuple of three numbers R, G and B values
                            or tuple of four numbers R, G, B and Alpha values
        """
        if isinstance(_x, tuple):
            super().__init__(_x[:3])
            if len(_x) > 3:
                self.a = _x[-1]
            else:
                self.a = self.peaks[1]  # type: ignore # pylance's complaining
        else:
            super().__init__(_x)



class RGBS(RGBNoAlpha[float], ForceFloat):
    """RGB colourspace in range 0.0 - 1.0"""

    peaks = (0., 1.)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tup3[float], /) -> None:
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[float]) -> None:
        super().__init__(_x)


class RGBAS(RGBAlpha[float], ForceFloat):
    """RGB with alpha colourspace in range 0.0 - 1.0"""

    peaks = (0., 1.)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tup3[float], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tup4[float], /) -> None:
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[float] | Tup4[float]) -> None:
        super().__init__(_x)


class RGB(RGBNoAlpha[int], ForceInt):
    """RGB colourspace in range 0 - 255"""
    peaks = (0, (2 ** 8) - 1)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tup3[int], /) -> None:
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[int]) -> None:
        super().__init__(_x)


class RGB24(RGB):
    """RGB colourspace in range 0 - 255"""
    ...


class RGB30(RGB):
    """RGB colourspace in range 0 - 1023"""
    peaks = (0, (2 ** 10) - 1)


class RGB36(RGB):
    """RGB colourspace in range 0 - 4095"""
    peaks = (0, (2 ** 12) - 1)


class RGB42(RGB):
    """RGB colourspace in range 0 - 16383"""
    peaks = (0, (2 ** 14) - 1)


class RGB48(RGB):
    """RGB colourspace in range 0 - 65535"""
    peaks = (0, (2 ** 16) - 1)


class RGBA(RGBAlpha[int], ForceInt):
    """RGB with alpha colourspace in range 0 - 255"""

    peaks = (0, (2 ** 8) - 1)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tup3[int], /) -> None:
        ...

    @overload
    def __init__(self, _x: Tup4[int], /) -> None:
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[int] | Tup4[int]) -> None:
        super().__init__(_x)


class RGBA32(RGBA):
    """RGB with alpha colourspace in range 0 - 255"""
    ...


class RGBA40(RGBA):
    """RGB with alpha colourspace in range 0 - 1023"""
    peaks = (0, (2 ** 10) - 1)


class RGBA48(RGBA):
    """RGB with alpha colourspace in range 0 - 4095"""
    peaks = (0, (2 ** 12) - 1)


class RGBA56(RGBA):
    """RGB with alpha colourspace in range 0 - 16383"""
    peaks = (0, (2 ** 14) - 1)


class RGBA64(RGBA):
    """RGB with alpha colourspace in range 0 - 65535"""
    peaks = (0, (2 ** 16) - 1)


class HueSaturationBased(ForceFloat, ColourSpace[float], ABC):
    """Base class for Hue and Saturation based colourspace"""

    h: float
    """Hue value"""

    s: float
    """Saturation"""

    peaks = (0., 1.)

    @abstractmethod
    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[float]) -> None:
        super().__init__()

    def to_xyz(self) -> XYZ:
        return self.to_rgb(RGBS).to_xyz()

    def to_xyy(self) -> xyY:
        return self.to_xyz().to_xyy()

    def to_lab(self) -> Lab:
        return self.to_xyz().to_lab()

    def to_lch_ab(self) -> LCHab:
        return self.to_xyz().to_lch_ab()

    def to_luv(self) -> Luv:
        return self.to_xyz().to_luv()

    def to_lch_uv(self) -> LCHuv:
        return self.to_xyz().to_lch_uv()

    def to_html(self) -> HTML:
        return self.to_rgb(RGB).to_html()

    def to_ass_color(self) -> ASSColor:
        return self.to_rgb(RGB).to_ass_color()

    @classmethod
    @check_annotations
    def from_ass_val(cls: Type[THSX], _x: Tup3[Nb8bit]) -> THSX:
        """
        Make a Hue Saturation based object from ASS values

        :param _x:          Tuple of integer in the range 0 - 255
        :return:            Hue Saturation based object
        """
        return cls(cast(Tup3[float], tuple(x / 255 for x in _x)))

    def to_ass_val(self) -> Tup3[float]:
        """
        Make a tuple of float of this current object in range 0 - 255

        :return:            Tuple of integer in the range 0 - 255
        """
        return cast(Tup3[float], tuple(round(x * 255) for x in self))

    def as_chromatic_circle(self) -> Tup3[float]:
        """
        Change H in a chromatic circle in range 0.0 - 360.0

        :return:            Tuple of float with H in range 0.0 - 360.0
        """
        return cast(Tup3[float], (self.h * 360, *(*self, )[1:3]))


class HSL(HueSaturationBased):
    """HSL colourspace in range 0.0 - 1.0"""

    l: float
    """Lightness value"""

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> HSL:
        """
        Make a new HSL colourspace object

        :param _x:          Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tup3[float]) -> HSL:
        """
        Make a new HSL colourspace object

        :param _x:          Tuple of three numbers H, S and L values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tup3[float]) -> HSL:
        return _x.to_hsl() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a new HSL colourspace object

        :param _x:          Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tup3[float]) -> None:
        """
        Make a new HSL colourspace object

        :param _x:          Tuple of three numbers H, S and L values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[float]) -> None:
        super().__init__(_x)
        if isinstance(_x, tuple):
            self.h, self.s, self.l = _x

    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        return RGBS(CC.hsl_to_rgb(*self)).to_rgb(rgb_type)

    def to_hsl(self) -> HSL:
        return self

    def to_hsv(self) -> HSV:
        return self.to_rgb(RGBS).to_hsv()


class HSV(HueSaturationBased):
    """HSV colourspace in range 0.0 - 1.0"""

    v: float
    """Value value"""

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> HSV:
        """
        Make a new HSV colourspace object

        :param _x:          Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tup3[float]) -> HSV:
        """
        Make a new HSL colourspace object

        :param _x:          Tuple of three numbers H, S and V values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tup3[float]) -> HSV:
        return _x.to_hsv() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        """
        Make a new HSV colourspace object

        :param _x:          Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tup3[float], /) -> None:
        """
        Make a new HSL colourspace object

        :param _x:          Tuple of three numbers H, S and V values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[float]) -> None:
        super().__init__(_x)
        if isinstance(_x, tuple):
            self.h, self.s, self.v = _x

    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        return RGBS(CC.hsv_to_rgb(*self)).to_rgb(rgb_type)

    def to_hsl(self) -> HSL:
        return self.to_rgb(RGBS).to_hsl()

    def to_hsv(self) -> HSV:
        return self


class Opacity(ColourSpace[float]):
    """Opacity colourspace like in range 0.0 - 1.0"""

    value: float
    """Value in floating format in the range 0.0 - 1.0"""

    data: str
    """ASS value, hexadecimal inversed"""

    def __init__(self, _x: Pct) -> None:
        """
        Make an Opacity colourspace object

        :param _x:      Percentage of the opacity.
                        1.0 means full opaque
                        0.0 means full transparent
        """
        super().__init__()
        self.value = clamp_value(_x, 0., 1.0)
        self.data = f'&H{round(abs(self.value * 255 - 255)):02X}&'

    @overload
    @classmethod
    def from_ass_val(cls, _x: str) -> Opacity:
        """
        Convert an ASS string value to an Opacity object

        :param _x:          ASS alpha string in the format &HXX&
        :return:            Opacity object
        """
        ...

    @overload
    @classmethod
    def from_ass_val(cls, _x: Nb8bit) -> Opacity:
        """
        Convert an ASS integer value to an Opacity object

        :param _x:          ASS integer string in the range 0 - 255
        :return:            Opacity object
        """
        ...

    @classmethod
    def from_ass_val(cls, _x: str | Nb8bit) -> Opacity:
        if isinstance(_x, str):
            if match := re.fullmatch(r"&H([0-9A-F]{2})&", _x):
                x = float(int(match.group(1), 16))
            else:
                raise ValueError(
                    f'Opacity: Provided ASS alpha string {_x} is not in the expected format &HXX&'
                )
        else:
            x = float(_x)
        x = (255 - x) / 255
        return cls(x)

    def __str__(self) -> str:
        return self.data

    def __repr__(self) -> str:
        return repr(self.value)

    def interpolate(self, nobj: TCS, pct: Pct, /) -> TCS:
        return cast(
            TCS,
            Opacity(
                self._colour_values['value'] * (1 - pct) + nobj._colour_values['value'] * pct
            )
        )

    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        raise NotImplementedError

    def to_xyz(self) -> XYZ:
        raise NotImplementedError

    def to_xyy(self) -> xyY:
        raise NotImplementedError

    def to_lab(self) -> Lab:
        raise NotImplementedError

    def to_lch_ab(self) -> LCHab:
        raise NotImplementedError

    def to_luv(self) -> Luv:
        raise NotImplementedError

    def to_lch_uv(self) -> LCHuv:
        raise NotImplementedError

    def to_hsl(self) -> HSL:
        raise NotImplementedError

    def to_hsv(self) -> HSV:
        raise NotImplementedError

    def to_html(self) -> HTML:
        raise NotImplementedError

    def to_ass_color(self) -> ASSColor:
        raise NotImplementedError


class HexBased(ColourSpace[str], ABC):
    """Hexadecimal based colourspace"""

    _rgb: RGB
    """Internal RGB object corresponding to the hexadecimal value"""

    data: str
    """Hexadecimal value"""

    def __str__(self) -> str:
        return self.data

    def __repr__(self) -> str:
        return repr(self.data)

    def interpolate(self, nobj: TCS, pct: Pct, /) -> TCS:
        return cast(TCS, self.__class__(self._rgb.interpolate(nobj.to_rgb(RGB), pct)))

    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        return self._rgb.to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return self.to_rgb(RGBS).to_xyz()

    def to_xyy(self) -> xyY:
        return self.to_xyz().to_xyy()

    def to_lab(self) -> Lab:
        return self.to_xyz().to_lab()

    def to_lch_ab(self) -> LCHab:
        return self.to_xyz().to_lch_ab()

    def to_luv(self) -> Luv:
        return self.to_xyz().to_luv()

    def to_lch_uv(self) -> LCHuv:
        return self.to_xyz().to_lch_uv()

    def to_hsl(self) -> HSL:
        return self.to_rgb(RGBS).to_hsl()

    def to_hsv(self) -> HSV:
        return self.to_rgb(RGBS).to_hsv()

    @staticmethod
    def hex_to_int(h: str) -> int:
        """
        Convert hexadecimal to based 10 integer

        :param h:       Hexadecimal value
        :return:        Base 10 value
        """
        return int(h, 16)


class HTML(HexBased):
    """HTML colourspace object"""

    _rgb: RGB
    data: str

    @overload
    def __new__(cls, _x: str) -> HTML:
        """
        Make a HTML colourspace object

        :param _x:      HTML string
        """
        ...

    @overload
    def __new__(cls, _x: Tup3Str) -> HTML:
        """
        Make a HTML colourspace object

        :param _x:      Tuple of three hexadecimal values
        """
        ...

    @overload
    def __new__(cls, _x: Tup3[int]) -> HTML:
        """
        Make a HTML colourspace object

        :param _x:      Tuple of three bases 10 values
        """
        ...

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> HTML:
        """
        Make a HTML colourspace object

        :param _x:      Colourspace object
        """
        ...

    def __new__(cls, _x: str | Tup3Str | Tup3[int] | ColourSpace[TCV_co]) -> HTML:
        return _x.to_html() if not isinstance(_x, (str, tuple)) else super().__new__(cls)

    @overload
    def __init__(self, _x: str) -> None:
        """
        Make a HTML colourspace object

        :param _x:      HTML string
        """
        ...

    @overload
    def __init__(self, _x: Tup3Str) -> None:
        """
        Make a HTML colourspace object

        :param _x:      Tuple of three hexadecimal values
        """
        ...

    @overload
    def __init__(self, _x: Tup3[int]) -> None:
        """
        Make a HTML colourspace object

        :param _x:      Tuple of three bases 10 values
        """
        ...

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a HTML colourspace object

        :param _x:      Colourspace object
        """
        ...

    def __init__(self, _x: str | Tup3Str | Tup3[int] | ColourSpace[TCV_co]) -> None:
        super().__init__()
        if isinstance(_x, (str, tuple)):
            if isinstance(_x, str):
                seq = _x
                match = re.fullmatch(r"#([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})", _x)
                if match:
                    r, g, b = map(self.hex_to_int, match.groups())
                    self._rgb = RGB((r, g, b))
                else:
                    ValueError('No match found')
            elif isinstance(_x, tuple):
                if all(isinstance(x, int) for x in _x):
                    _x = cast(Tup3[int], _x)
                    self._rgb = RGB(_x)
                    seq = ''.join(hex(x)[2:].zfill(2) for x in self._rgb)
                else:
                    _x = cast(Tup3Str, _x)
                    r, g, b = map(self.hex_to_int, _x)
                    self._rgb = RGB((r, g, b))
                    seq = ''.join(_x)

            self.data = "#" + seq.upper()

    def to_html(self) -> HTML:
        return self

    def to_ass_color(self) -> ASSColor:
        return self._rgb.to_ass_color()


class ASSColor(HexBased):
    """AssColor colourspace object"""

    _rgb: RGB
    data: str

    @overload
    def __new__(cls, _x: str) -> ASSColor:
        ...

    @overload
    def __new__(cls, _x: Tup3Str) -> ASSColor:
        ...

    @overload
    def __new__(cls, _x: Tup3[int]) -> ASSColor:
        ...

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> ASSColor:
        ...

    def __new__(cls, _x: str | Tup3Str | Tup3[int] | ColourSpace[TCV_co]) -> ASSColor:
        return _x.to_ass_color() if not isinstance(_x, (str, tuple)) else super().__new__(cls)

    @overload
    def __init__(self, _x: str) -> None:
        """
        Make a AssColor object from ASS string of the form "&HBBGGRR&"

        .. code-block:: python

            >>> ASSColor('&HF1D410&')

        :param _x:      ASS string
        """
        ...

    @overload
    def __init__(self, _x: Tup3Str) -> None:
        """
        Make a AssColor object from a tuple of string of the form ('BB', 'GG', 'RR')

        .. code-block:: python

            >>> ASSColor(('F1', 'D4', '10'))

        :param _x:      Tuple of string
        """
        ...

    @overload
    def __init__(self, _x: Tup3[int]) -> None:
        """
        Make a AssColor object from a tuple of int of the form (BB, GG, RR)

        .. code-block:: python

            >>> ASSColor((241, 212, 16))
            >>> ASSColor((0xF1, 0xD4, 0x10))

        :param _x:      Tuple of integers
        """
        ...

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a AssColor object from an other ColourSpace object

        :param _x:      ColourSpace object
        """
        ...

    def __init__(self, _x: str | Tup3Str | Tup3[int] | ColourSpace[TCV_co]) -> None:
        super().__init__()
        if isinstance(_x, (str, tuple)):
            if isinstance(_x, str):
                seq = _x[2:-1]
                if match := re.fullmatch(r"&H([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})&", _x):
                    r, g, b = map(self.hex_to_int, match.groups()[::-1])
                    self._rgb = RGB((r, g, b))
                else:
                    ValueError('No match found')
            elif isinstance(_x, tuple):
                if all(isinstance(x, int) for x in _x):
                    self._rgb = RGB(cast(Tup3[int], _x[::-1]))
                    seq = ''.join(hex(cast(int, x))[2:].zfill(2) for x in _x)
                else:
                    _x = cast(Tup3Str, _x)
                    r, g, b = map(self.hex_to_int, _x)
                    self._rgb = RGB((r, g, b))
                    seq = ''.join(_x)

            self.data = "&H" + seq.upper() + "&"

    def to_html(self) -> HTML:
        return self._rgb.to_html()

    def to_ass_color(self) -> ASSColor:
        return self


class XYZBased(ForceFloat, ColourSpace[float], ABC):
    """Base colourspace class for colourspace where the conversions need XYZ"""

    def to_hsl(self) -> HSL:
        return self.to_rgb(RGBS).to_hsl()

    def to_hsv(self) -> HSV:
        return self.to_rgb(RGBS).to_hsv()

    def to_html(self) -> HTML:
        return self.to_rgb(RGB).to_html()

    def to_ass_color(self) -> ASSColor:
        return self.to_rgb(RGB).to_ass_color()


class XYZ(XYZBased):
    """XYZ colourspace object"""

    x: float
    """Mix of the three CIE RGB curves chosen to be nonnegative value"""

    y: float
    """Luminance value"""

    z: float
    """Quasi-equal to blue value"""

    peaks = (0, 1.)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co], /) -> XYZ:
        """
        Make a XYZ colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tup3[float], /) -> XYZ:
        """
        Make a XYZ colourspace object

        :param _x:      A tuple of three values in the range 0.0 - 1.0
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tup3[float]) -> XYZ:
        return _x.to_xyz() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co], /) -> None:
        """
        Make a XYZ colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tup3[float], /) -> None:
        """
        Make a XYZ colourspace object

        :param _x:      A tuple of three values in the range 0.0 - 1.0
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.x, self.y, self.z = _x

    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        return RGBS(CC.xyz_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return self

    def to_xyy(self) -> xyY:
        return xyY(CC.xyz_to_xyy(*self))

    def to_lab(self) -> Lab:
        return Lab(CC.xyz_to_lab(*self))

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.xyz_to_lch_ab(*self))

    def to_luv(self) -> Luv:
        return Luv(CC.xyz_to_luv(*self))

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.xyz_to_lch_uv(*self))


class xyY(XYZBased):
    """xyY colourspace object"""
    x: float
    y: float
    Y: float

    peaks = (0, 1.)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> xyY:
        """
        Make a xyY colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tup3[float]) -> xyY:
        """
        Make a xyY colourspace object

        :param _x:      A tuple of three values in the range 0.0 - 1.0
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tup3[float]) -> xyY:
        return _x.to_xyy() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a xyY colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tup3[float]) -> None:
        """
        Make a xyY colourspace object

        :param _x:      A tuple of three values in the range 0.0 - 1.0
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.x, self.y, self.z = _x

    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        return RGBS(CC.xyy_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.xyy_to_xyz(*self))

    def to_xyy(self) -> xyY:
        return self

    def to_lab(self) -> Lab:
        return Lab(CC.xyy_to_lab(*self))

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.xyy_to_lch_ab(*self))

    def to_luv(self) -> Luv:
        return Luv(CC.xyy_to_luv(*self))

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.xyy_to_lch_uv(*self))


class Lab(XYZBased):
    """Lab colourspace object based on Cartesian coordinates"""
    L: float
    """Lightness value"""
    a: float
    """
    Relative to the green–red opponent colors,
    with negative values toward green and positive values toward red
    """
    b: float
    """
    The b* axis represents the blue–yellow opponents,
    with negative numbers toward blue and positive toward yellow
    """

    peaks = (-50000., 50000)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> Lab:
        """
        Make a Lab colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tup3[float]) -> Lab:
        """
        Make a Lab colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tup3[float]) -> Lab:
        return _x.to_lab() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a Lab colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tup3[float]) -> None:
        """
        Make a Lab colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.L, self.a, self.b = _x

    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        return RGBS(CC.lab_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.lab_to_xyz(*self))

    def to_xyy(self) -> xyY:
        return xyY(CC.lab_to_xyy(*self))

    def to_lab(self) -> Lab:
        return self

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.lab_to_lch_ab(*self))

    def to_luv(self) -> Luv:
        return Luv(CC.lab_to_luv(*self))

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.lab_to_lch_uv(*self))


class LCHab(XYZBased):
    """
    LCHab colourspace object based on polar coordinates
    Cylindrical model of the Lab colourspace
    """

    L: float
    """Lightness value"""

    C: float
    """Chroma, relative saturation"""

    H: float
    """Hue angle, angle of the hue in the CIELAB color wheel"""

    peaks = (-50000., 50000)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> LCHab:
        """
        Make a LCHab colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tup3[float]) -> LCHab:
        """
        Make a LCHab colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tup3[float]) -> LCHab:
        return _x.to_lch_ab() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a LCHab colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tup3[float]) -> None:
        """
        Make a LCHab colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.L, self.C, self.H = _x

    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        return RGBS(CC.lch_ab_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.lch_ab_to_xyz(*self))

    def to_xyy(self) -> xyY:
        return xyY(CC.lch_ab_to_xyy(*self))

    def to_lab(self) -> Lab:
        return Lab(CC.lch_ab_to_lab(*self))

    def to_lch_ab(self) -> LCHab:
        return self

    def to_luv(self) -> Luv:
        return Luv(CC.lch_ab_to_luv(*self))

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.lch_ab_to_lch_uv(*self))


class Luv(XYZBased):
    """Luv colourspace object based on Cartesian coordinates"""

    L: float
    """Lightness value"""
    u: float
    v: float

    peaks = (-50000., 50000)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> Luv:
        """
        Make a Luv colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tup3[float]) -> Luv:
        """
        Make a Luv colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tup3[float]) -> Luv:
        return _x.to_luv() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a Luv colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tup3[float]) -> None:
        """
        Make a Luv colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.L, self.u, self.v = _x

    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        return RGBS(CC.luv_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.luv_to_xyz(*self))

    def to_xyy(self) -> xyY:
        return xyY(CC.luv_to_xyy(*self))

    def to_lab(self) -> Lab:
        return Lab(CC.luv_to_lab(*self))

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.luv_to_lch_ab(*self))

    def to_luv(self) -> Luv:
        return self

    def to_lch_uv(self) -> LCHuv:
        return LCHuv(CC.luv_to_lch_uv(*self))


class LCHuv(XYZBased):
    """
    LCHab colourspace object based on polar coordinates
    Cylindrical model of the Luv colourspace
    """

    L: float
    """Lightness value"""

    C: float
    """Chroma, relative saturation"""

    H: float
    """Hue angle, angle of the hue in the CIELAB color wheel"""

    peaks = (-50000., 50000)

    @overload
    def __new__(cls, _x: ColourSpace[TCV_co]) -> LCHuv:
        """
        Make a LCHuv colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __new__(cls, _x: Tup3[float]) -> LCHuv:
        """
        Make a LCHuv colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __new__(cls, _x: ColourSpace[TCV_co] | Tup3[float]) -> LCHuv:
        return _x.to_lch_uv() if not isinstance(_x, tuple) else super().__new__(cls)

    @overload
    def __init__(self, _x: ColourSpace[TCV_co]) -> None:
        """
        Make a LCHuv colourspace object

        :param _x:      Colourspace object
        """
        ...

    @overload
    def __init__(self, _x: Tup3[float]) -> None:
        """
        Make a LCHuv colourspace object

        :param _x:      A tuple of three values
        """
        ...

    def __init__(self, _x: ColourSpace[TCV_co] | Tup3[float]) -> None:
        super().__init__()
        if isinstance(_x, tuple):
            self.L, self.C, self.H = _x

    def to_rgb(self, rgb_type: Type[TRGB], /) -> TRGB:
        return RGBS(CC.lch_uv_to_rgb(*self)).to_rgb(rgb_type)

    def to_xyz(self) -> XYZ:
        return XYZ(CC.lch_uv_to_xyz(*self))

    def to_xyy(self) -> xyY:
        return xyY(CC.lch_uv_to_xyy(*self))

    def to_lab(self) -> Lab:
        return Lab(CC.lch_uv_to_lab(*self))

    def to_lch_ab(self) -> LCHab:
        return LCHab(CC.lch_uv_to_lch_ab(*self))

    def to_luv(self) -> Luv:
        return Luv(CC.lch_uv_to_luv(*self))

    def to_lch_uv(self) -> LCHuv:
        return self
