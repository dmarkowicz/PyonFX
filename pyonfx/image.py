from __future__ import annotations

__all__ = ['Image']

from pathlib import Path
from typing import List, NoReturn

import cv2

from .colourspace import ASSColor, Opacity
from .shape import Pixel
from .types import AnyPath


class Image:
    path: Path

    def __init__(self, image: AnyPath) -> None:
        self.path = Path(image)

    def to_ass(self) -> NoReturn:
        raise NotImplementedError

    def to_pixels(self) -> List[Pixel]:
        """
        Convert current image file to a list of Pixel
        It is strongly recommended to create a dedicated style for pixels,
        thus, you will write less tags for line in your pixels,
        which means less size for your .ass file.

        Style suggested as an=7, bord=0, shad=0

        :return:            List of Pixel
        """
        img_bgr = cv2.imread(str(self.path))
        rows, columns, channels = img_bgr.shape
        return [
            Pixel(
                co, ro,
                Opacity(1.0),
                ASSColor(tuple(map(int, (img_bgr[ro, co, ch] for ch in range(channels)))))  # type: ignore
            )
            for ro in range(rows)
            for co in range(columns)
        ]
