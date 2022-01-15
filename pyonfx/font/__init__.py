# flake8: noqa
__all__ = ['Font', 'get_font']

import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if sys.platform == 'win32':
    from ._windows import Font
elif sys.platform in ['linux', 'darwin'] and 'sphinx' not in sys.modules:
    from ._linux_macos import Font
else:
    raise NotImplementedError

if TYPE_CHECKING:
    from ..core import Style
else:
    Style = Any

@lru_cache(maxsize=None)
def get_font(style: Style) -> Font:
    return Font(style)
