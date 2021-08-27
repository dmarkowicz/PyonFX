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
"""
This module contains the Font class definition, which has some functions
to help getting informations from a specific font
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Dict, Final, Tuple

from .shape import Shape

if sys.platform == "win32":
    import win32con
    import win32gui
    import win32ui
    from win32helper.win32typing import PyCFont
elif sys.platform in ["linux", "darwin"] and "sphinx" not in sys.modules:
    import cairo
    import gi

    gi.require_version("Pango", "1.0")
    gi.require_version("PangoCairo", "1.0")

    import html

    from gi.repository import Pango, PangoCairo

if TYPE_CHECKING:
    from .core import Style

# CONFIGURATION
FONT_PRECISION: Final[int] = 64
"""Font scale for better precision output from native font system"""
LIBASS_FONTHACK: Final[bool] = True
"""Scale font data to fontsize? (no effect on windows)"""
PANGO_SCALE: Final[int] = 1024
"""The PANGO_SCALE macro represents the scale between dimensions used for Pango distances and device units."""


class Font:
    """
    Font class definition
    """
    style: Style
    xscale: float
    yscale: float
    hspace: float
    upscale = FONT_PRECISION
    downscale = 1 / FONT_PRECISION

    metrics: Dict[str, float]
    pycfont: PyCFont

    def __init__(self, style: Style) -> None:
        self.style = style
        self.xscale = style.scale_x / 100
        self.yscale = style.scale_y / 100
        self.hspace = style.spacing

        if sys.platform == "win32":
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
            # self.metrics = win32gui.GetTextMetrics(self.dc)
            self.metrics = win32gui.GetTextMetrics()
        elif sys.platform in {"linux", "darwin"}:
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
            self.metrics = Pango.Context.get_metrics(
                self.layout.get_context(), self.layout.get_font_description()
            )

            if LIBASS_FONTHACK:
                self.fonthack_scale = self.style.fontsize / (
                    (self.metrics.get_ascent() + self.metrics.get_descent())
                    / PANGO_SCALE
                    * self.downscale
                )
            else:
                self.fonthack_scale = 1
        else:
            raise NotImplementedError

    def __del__(self) -> None:
        if sys.platform == "win32":
            win32gui.DeleteObject(self.pycfont.GetSafeHandle())
            win32gui.DeleteDC(self.dc)

    def get_metrics(self) -> Tuple[float, float, float, float]:
        if sys.platform == "win32":
            const = self.downscale * self.yscale
            return (
                # 'height': self.metrics['Height'] * const,
                self.metrics["Ascent"] * const,
                self.metrics["Descent"] * const,
                self.metrics["InternalLeading"] * const,
                self.metrics["ExternalLeading"] * const,
            )
        elif sys.platform in {"linux", "darwin"}:
            const = self.downscale * self.yscale * self.fonthack_scale / PANGO_SCALE
            return (
                # 'height': (self.metrics.get_ascent() + self.metrics.get_descent()) * const,
                self.metrics.get_ascent() * const,
                self.metrics.get_descent() * const,
                0.0,
                self.layout.get_spacing() * const,
            )
        else:
            raise NotImplementedError

    def get_text_extents(self, text: str) -> Tuple[float, float]:
        if sys.platform == "win32":
            cx, cy = win32gui.GetTextExtentPoint32(self.dc, text)

            return (
                (cx * self.downscale + self.hspace * (len(text) - 1)) * self.xscale,
                cy * self.downscale * self.yscale,
            )
        elif sys.platform in {"linux", "darwin"}:
            if not text:
                return 0.0, 0.0

            def get_rect(new_text):
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
        else:
            raise NotImplementedError

    def text_to_shape(self, text: str) -> Shape:
        if sys.platform == "win32":
            # TODO: Calcultating distance between origins of character cells (just in case of spacing)

            # Add path to device context
            win32gui.BeginPath(self.dc)
            win32gui.ExtTextOut(self.dc, 0, 0, 0x0, None, text)  # type: ignore
            win32gui.EndPath(self.dc)
            # Getting Path produced by Microsoft API
            points, type_points = win32gui.GetPath(self.dc)

            # Checking for errors
            if len(points) == 0 or len(points) != len(type_points):
                raise RuntimeError(
                    'This should never happen: function win32gui.GetPath has returned something unexpected.'
                    '\nPlease report this to the developer'
                )

            # Defining variables
            shape, last_type = [], None
            mult_x, mult_y = self.downscale * self.xscale, self.downscale * self.yscale

            # Convert points to shape
            i = 0
            while i < len(points):
                cur_point, cur_type = points[i], type_points[i]

                if cur_type == win32con.PT_MOVETO:
                    if last_type != win32con.PT_MOVETO:
                        # Avoid repetition of command tags
                        shape.append("m")
                        last_type = cur_type
                    shape.extend(
                        [
                            Shape.format_value(cur_point[0] * mult_x),
                            Shape.format_value(cur_point[1] * mult_y),
                        ]
                    )
                    i += 1
                elif cur_type == win32con.PT_LINETO or cur_type == (
                    win32con.PT_LINETO | win32con.PT_CLOSEFIGURE
                ):
                    if last_type != win32con.PT_LINETO:
                        # Avoid repetition of command tags
                        shape.append("l")
                        last_type = cur_type
                    shape.extend(
                        [
                            Shape.format_value(cur_point[0] * mult_x),
                            Shape.format_value(cur_point[1] * mult_y),
                        ]
                    )
                    i += 1
                elif cur_type == win32con.PT_BEZIERTO or cur_type == (
                    win32con.PT_BEZIERTO | win32con.PT_CLOSEFIGURE
                ):
                    if last_type != win32con.PT_BEZIERTO:
                        # Avoid repetition of command tags
                        shape.append("b")
                        last_type = cur_type
                    shape.extend(
                        [
                            Shape.format_value(cur_point[0] * mult_x),
                            Shape.format_value(cur_point[1] * mult_y),
                            Shape.format_value(points[i + 1][0] * mult_x),
                            Shape.format_value(points[i + 1][1] * mult_y),
                            Shape.format_value(points[i + 2][0] * mult_x),
                            Shape.format_value(points[i + 2][1] * mult_y),
                        ]
                    )
                    i += 3
                else:  # If there is an invalid type -> skip, for safeness
                    i += 1

            # Clear device context path
            win32gui.AbortPath(self.dc)

            return Shape(" ".join(shape))
        elif sys.platform in {"linux", "darwin"}:
            # Defining variables
            shape, last_type = [], None

            def shape_from_text(new_text, x_add):
                nonlocal shape, last_type

                self.layout.set_markup(
                    f"<span "
                    f'strikethrough="{str(self.style.strikeout).lower()}" '
                    f'underline="{"single" if self.style.underline else "none"}"'
                    f">"
                    f"{html.escape(new_text)}"
                    f"</span>",
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
                for current_entry in path:
                    current_type = current_entry[0]
                    current_path = current_entry[1]

                    if current_type == 0:  # MOVE_TO
                        if last_type != current_type:
                            # Avoid repetition of command tags
                            shape.append("m")
                            last_type = current_type
                        shape.extend(
                            [
                                Shape.format_value(current_path[0] + x_add),
                                Shape.format_value(current_path[1]),
                            ]
                        )
                    elif current_type == 1:  # LINE_TO
                        if last_type != current_type:
                            # Avoid repetition of command tags
                            shape.append("l")
                            last_type = current_type
                        shape.extend(
                            [
                                Shape.format_value(current_path[0] + x_add),
                                Shape.format_value(current_path[1]),
                            ]
                        )
                    elif current_type == 2:  # CURVE_TO
                        if last_type != current_type:
                            # Avoid repetition of command tags
                            shape.append("b")
                            last_type = current_type
                        shape.extend(
                            [
                                Shape.format_value(current_path[0] + x_add),
                                Shape.format_value(current_path[1]),
                                Shape.format_value(current_path[2] + x_add),
                                Shape.format_value(current_path[3]),
                                Shape.format_value(current_path[4] + x_add),
                                Shape.format_value(current_path[5]),
                            ]
                        )

                self.context.new_path()

            curr_width = 0

            for i, char in enumerate(text):
                shape_from_text(char, curr_width + self.hspace * self.xscale * i)
                curr_width += self.get_text_extents(char)[0]

            return Shape(" ".join(shape))
        else:
            raise NotImplementedError
