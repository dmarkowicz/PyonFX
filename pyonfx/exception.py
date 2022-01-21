

class _StringRepresentable(BaseException):
    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__class__.__name__


class MatchNotFoundError(LookupError, _StringRepresentable):
    """Match object not found"""


class LineNotFoundWarning(SyntaxWarning, _StringRepresentable):
    """Line in ASS not found"""
