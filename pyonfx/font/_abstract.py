"""Font getting data module"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from ..core import Style
    from ..shape import Shape
else:
    Shape, Style = Any, Any

FONT_PRECISION = 64
"""Font scale for better precision output from native font system"""


class _Metrics(NamedTuple):
    ascent: float
    descent: float
    internal_leading: float
    external_leading: float


class _TextExtents(NamedTuple):
    width: float
    height: float


class _AbstractFont(ABC):
    """Class for getting data from fonts"""
    style: Style
    xscale: float
    yscale: float
    hspace: float
    upscale = FONT_PRECISION
    downscale = 1 / FONT_PRECISION

    def __init__(self, style: Style) -> None:
        """
        Initialise a font object

        :param style:       Style object from core module
        """
        self.style = style
        self.xscale = style.scale_x / 100
        self.yscale = style.scale_y / 100
        self.hspace = style.spacing

    @abstractmethod
    def __del__(self) -> None:
        ...

    @property
    @abstractmethod
    def metrics(self) -> _Metrics:
        """
        Get metrics of the current font

        :return:            A tuple containing text data in this order:
                            (ascent, descent, internal_leading, external_leading)
        """
        ...

    @abstractmethod
    def text_extents(self, text: str) -> _TextExtents:
        """
        Get text extents of the specified text

        :param text:        Text in string
        :return:            A tuple containing text data in this order:
                            (width, height)
        """
        ...

    @abstractmethod
    def text_to_shape(self, text: str) -> Shape:
        """
        Convert specified text string into a Shape object

        :param text:        Text in string
        :return:            A Shape object
        """
        ...
