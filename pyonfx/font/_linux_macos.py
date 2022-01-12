
import html
from typing import TYPE_CHECKING, Any, List, Tuple

import cairo  # type: ignore
import gi  # type: ignore
gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo  # type: ignore # noqa E402

from ..shape import DrawingCommand, DrawingProp, Shape  # noqa E402

if TYPE_CHECKING:
    from ..core import Style
else:
    Style = Any

from ._abstract import _Font  # noqa E402

LIBASS_FONTHACK = True
"""Scale font data to fontsize?"""
PANGO_SCALE = 1024
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
