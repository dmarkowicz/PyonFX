# PyonFX: An easy way to create KFX (Karaoke Effects) and complex typesetting using the ASS format (Advanced Substation Alpha).
# Copyright (C) 2019 Antonio Strippoli (CoffeeStraw/YellowFlash)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyonFX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.
"""Font getting data module"""
from __future__ import annotations

__all__ = ['Font']

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Final, List, Tuple

from .shape import DrawingCommand, DrawingProp, Shape

if TYPE_CHECKING:
    from .core import Style

FONT_PRECISION: Final[int] = 64
"""Font scale for better precision output from native font system"""


class _Font(ABC):
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
    def metrics(self) -> Tuple[float, float, float, float]:
        """
        Get metrics of the current font

        :return:            A tuple containing text data in this order:
                            (ascent, descent, internal_leading, external_leading)
        """
        ...

    @abstractmethod
    def get_text_extents(self, text: str) -> Tuple[float, float]:
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


if sys.platform == "win32":
    import win32con
    import win32gui
    import win32ui
    from win32helper.win32typing import PyCFont  # type: ignore

    class Font(_Font):
        _metrics: Dict[str, float]

        pycfont: PyCFont

        def __init__(self, style: Style) -> None:
            super().__init__(style)
            # Create device context
            self.dc = win32gui.CreateCompatibleDC(None)
            # Set context coordinates mapping mode
            win32gui.SetMapMode(self.dc, win32con.MM_TEXT)
            # Set context backgrounds to transparent
            win32gui.SetBkMode(self.dc, win32con.TRANSPARENT)
            # Create font handle
            font_spec: Dict[str, Any] = {
                "height": int(self.style.fontsize * self.upscale),
                "width": 0,
                "escapement": 0,
                "orientation": 0,
                "weight": win32con.FW_BOLD if self.style.bold else win32con.FW_NORMAL,
                "italic": int(self.style.italic),
                "underline": int(self.style.underline),
                "strike out": int(self.style.strikeout),
                "charset": win32con.DEFAULT_CHARSET,
                "out precision": win32con.OUT_TT_PRECIS,
                "clip precision": win32con.CLIP_DEFAULT_PRECIS,
                "quality": win32con.ANTIALIASED_QUALITY,
                "pitch and family": win32con.DEFAULT_PITCH + win32con.FF_DONTCARE,
                "name": self.style.fontname,
            }
            self.pycfont = win32ui.CreateFont(font_spec)
            win32gui.SelectObject(self.dc, self.pycfont.GetSafeHandle())
            # Calculate metrics
            self._metrics = win32gui.GetTextMetrics(self.dc)

        def __del__(self) -> None:
            win32gui.DeleteObject(self.pycfont.GetSafeHandle())
            win32gui.DeleteDC(self.dc)

        @property
        def metrics(self) -> Tuple[float, float, float, float]:
            const = self.downscale * self.yscale
            return (
                # 'height': self.metrics['Height'] * const,
                self._metrics["Ascent"] * const,
                self._metrics["Descent"] * const,
                self._metrics["InternalLeading"] * const,
                self._metrics["ExternalLeading"] * const,
            )

        def get_text_extents(self, text: str) -> Tuple[float, float]:
            cx, cy = win32gui.GetTextExtentPoint32(self.dc, text)

            return (
                (cx * self.downscale + self.hspace * (len(text) - 1)) * self.xscale,
                cy * self.downscale * self.yscale,
            )

        def text_to_shape(self, text: str) -> Shape:
            if not text:
                raise ValueError(f'{self.__class__.__name__}: Text is empty!')
            # TODO: Calcultating distance between origins of character cells (just in case of spacing)

            # Add path to device context
            win32gui.BeginPath(self.dc)
            win32gui.ExtTextOut(self.dc, 0, 0, 0x0, None, text)
            win32gui.EndPath(self.dc)
            # Getting Path produced by Microsoft API
            points, type_points = win32gui.GetPath(self.dc)

            # Checking for errors
            if len(points) == 0 or len(points) != len(type_points):
                raise RuntimeError(
                    f'{self.__class__.__name__}: no points detected or mismatch length between points and type_points'
                )

            # Defining variables
            PT_MOVE, PT_LINE, PT_BÉZIER = win32con.PT_MOVETO, win32con.PT_LINETO, win32con.PT_BEZIERTO
            PT_CLOSE = win32con.PT_CLOSEFIGURE
            PT_LINE_OR_CLOSE, PT_BÉZIER_OR_CLOSE = PT_LINE | PT_CLOSE, PT_BÉZIER | PT_CLOSE

            cmds: List[DrawingCommand] = []
            DC, DP = DrawingCommand, DrawingProp
            m, l, b = DP.MOVE, DP.LINE, DP.BÉZIER

            points_types = iter(zip(points, type_points))
            while True:
                try:
                    (x0, y0), ptype = next(points_types)
                except StopIteration:
                    break
                if ptype == PT_MOVE:
                    cmds.append(DC(m, (x0, y0)))
                elif ptype in {PT_LINE, PT_LINE_OR_CLOSE}:
                    cmds.append(DC(l, (x0, y0)))
                elif ptype in {PT_BÉZIER, PT_BÉZIER_OR_CLOSE}:
                    (x1, y1), _ = next(points_types)
                    (x2, y2), _ = next(points_types)
                    cmds.append(DC(b, (x0, y0), (x1, y1), (x2, y2)))
                else:
                    pass

            # Clear device context path
            win32gui.AbortPath(self.dc)

            shape = Shape(cmds)
            shape.scale(self.downscale * self.xscale, self.downscale * self.yscale)
            return shape


elif sys.platform in ["linux", "darwin"] and "sphinx" not in sys.modules:
    import html

    import cairo  # type: ignore
    import gi  # type: ignore
    gi.require_version("Pango", "1.0")
    gi.require_version("PangoCairo", "1.0")
    from gi.repository import Pango, PangoCairo  # type: ignore

    LIBASS_FONTHACK: Final[bool] = True
    """Scale font data to fontsize? (no effect on windows)"""
    PANGO_SCALE: Final[int] = 1024
    """The PANGO_SCALE macro represents the scale between dimensions used for Pango distances and device units."""


    class Font(_Font):
        _metrics: Any

        def __init__(self, style: Style) -> None:
            super().__init__(style)
            surface = cairo.ImageSurface(cairo.Format.A8, 1, 1)

            self.context = cairo.Context(surface)
            self.layout = PangoCairo.create_layout(self.context)

            font_description = Pango.FontDescription()
            font_description.set_family(self.style.fontname)
            font_description.set_absolute_size(self.style.fontsize * self.upscale * PANGO_SCALE)
            font_description.set_weight(
                Pango.Weight.BOLD if self.style.bold else Pango.Weight.NORMAL
            )
            font_description.set_style(
                Pango.Style.ITALIC if self.style.italic else Pango.Style.NORMAL
            )

            self.layout.set_font_description(font_description)
            self._metrics = Pango.Context.get_metrics(
                self.layout.get_context(), self.layout.get_font_description()
            )

            if LIBASS_FONTHACK:
                self.fonthack_scale = self.style.fontsize / (
                    (self._metrics.get_ascent() + self._metrics.get_descent())
                    / PANGO_SCALE
                    * self.downscale
                )
            else:
                self.fonthack_scale = 1

        def __del__(self) -> None:
            pass

        @property
        def metrics(self) -> Tuple[float, float, float, float]:
            const = self.downscale * self.yscale * self.fonthack_scale / PANGO_SCALE
            return (
                # 'height': (self.metrics.get_ascent() + self.metrics.get_descent()) * const,
                self._metrics.get_ascent() * const,
                self._metrics.get_descent() * const,
                0.0,
                self.layout.get_spacing() * const,
            )

        def get_text_extents(self, text: str) -> Tuple[float, float]:
            if not text:
                return 0.0, 0.0

            def get_rect(new_text: str) -> Any:
                self.layout.set_markup(
                    f"<span "
                    f'strikethrough="{str(self.style.strikeout).lower()}" '
                    f'underline="{"single" if self.style.underline else "none"}"'
                    f">"
                    f"{html.escape(new_text)}"
                    f"</span>",
                    -1,
                )
                return self.layout.get_pixel_extents()[1]

            width = 0
            for char in text:
                width += get_rect(char).width

            return (
                (
                    width * self.downscale * self.fonthack_scale
                    + self.hspace * (len(text) - 1)
                )
                * self.xscale,
                get_rect(text).height
                * self.downscale
                * self.yscale
                * self.fonthack_scale,
            )

        def text_to_shape(self, text: str) -> Shape:
            if not text:
                raise ValueError(f'{self.__class__.__name__}: Text is empty!')
            curr_width = 0.
            cmds: List[DrawingCommand] = []
            DC, DP = DrawingCommand, DrawingProp
            m, l, b = DP.MOVE, DP.LINE, DP.BÉZIER

            for i, char in enumerate(text):
                x_add = curr_width + self.hspace * self.xscale * i

                self.layout.set_markup(
                    "<span "
                    f'strikethrough="{str(self.style.strikeout).lower()}" '
                    f'underline="{"single" if self.style.underline else "none"}"'
                    ">"
                    f"{html.escape(char)}"
                    "</span>",
                    -1,
                )

                self.context.save()
                self.context.scale(
                    self.downscale * self.xscale * self.fonthack_scale,
                    self.downscale * self.yscale * self.fonthack_scale,
                )
                PangoCairo.layout_path(self.context, self.layout)
                self.context.restore()
                path = self.context.copy_path()

                # Convert points to shape
                for ptype, ppath in path:  # type: ignore[attr-defined]
                    if ptype == 0:
                        cmds.append(DC(m, (ppath[0] + x_add, ppath[1])))
                    elif ptype == 1:
                        cmds.append(DC(l, (ppath[0] + x_add, ppath[1])))
                    elif ptype == 2:
                        cmds.append(
                            DC(
                                b,
                                (ppath[0] + x_add, ppath[1]),
                                (ppath[2] + x_add, ppath[3]),
                                (ppath[4] + x_add, ppath[5])
                            )
                        )

                self.context.new_path()
                curr_width += self.get_text_extents(char)[0]

            return Shape(cmds)

else:
    raise NotImplementedError
