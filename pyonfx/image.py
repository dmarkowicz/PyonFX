from __future__ import annotations

__all__ = ['Image']

from typing import NoReturn


class Image:
    def image_to_ass(self) -> NoReturn:
        raise NotImplementedError

    def image_to_pixels(self) -> NoReturn:
        raise NotImplementedError
