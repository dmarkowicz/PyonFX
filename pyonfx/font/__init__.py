# flake8: noqa

import sys

if sys.platform == 'win32':
    from ._windows import Font
elif sys.platform in ['linux', 'darwin'] and 'sphinx' not in sys.modules:
    from ._linux_macos import Font
else:
    raise NotImplementedError

__all__ = ['Font']
