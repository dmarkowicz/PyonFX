from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from typing import (Annotated, Any, Callable, Collection, Generic, Iterable,
                    Iterator, Reversible, Sequence, Sized, Tuple, TypeVar, Union, cast, get_args,
                    get_origin, get_type_hints)

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
F = TypeVar('F', bound=Callable[..., Any])
TCV_co = TypeVar('TCV_co', bound=Union[float, int, str], covariant=True)  # Type Color Value covariant
TCV_inv = TypeVar('TCV_inv', bound=Union[float, int, str])  # Type Color Value invariant
ACV = Union[float, int, str]
Nb = TypeVar('Nb', bound=Union[float, int])  # Number
Tup3 = Tuple[Nb, Nb, Nb]
Tup4 = Tuple[Nb, Nb, Nb, Nb]
Tup3Str = Tuple[str, str, str]

BézierCoord = Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float]
]


class CheckAnnotated(Generic[T], ABC):
    @abstractmethod
    def check(self, val: T | Iterable[T], param_name: str) -> None:
        ...


class ValueRangeInclExcl(CheckAnnotated[float]):
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def check(self, val: float | Iterable[float], param_name: str) -> None:
        val = [val] if isinstance(val, float) else val
        for v in val:
            if not self.x < v <= self.y:
                raise ValueError(f'{param_name} "{v}" is not in the range ({self.x}, {self.y})')


class ValueRangeIncInc(CheckAnnotated[float]):
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def check(self, val: float | Iterable[float], param_name: str) -> None:
        val = [val] if isinstance(val, float) else val
        for v in val:
            if not self.x < v < self.y:
                raise ValueError(f'{param_name} "{v}" is not in the range ({self.x}, {self.y})')


Nb8bit = Annotated[int, ValueRangeInclExcl(0, 256)]
Nb16bit = Annotated[int, ValueRangeInclExcl(0, 65536)]
NbFloat = Annotated[float, ValueRangeIncInc(0.0, 1.0)]
Pct = Annotated[float, ValueRangeIncInc(0.0, 1.0)]
Alignment = Annotated[int, ValueRangeIncInc(0, 9)]


def check_annotations(func: F, /) -> F:

    def _check_hint(hint: Any, value: Any, param_name: str) -> None:
        if get_origin(hint) is Annotated:
            # hint_type, *hint_args = get_args(hint)
            _, *hint_args = get_args(hint)
            for hint_arg in hint_args:
                if isinstance(hint_arg, CheckAnnotated):
                    hint_arg.check(value, param_name)
                else:
                    raise TypeError

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        type_hints = get_type_hints(func, include_extras=True)
        for value, (param_name, hint) in zip(list(args) + list(kwargs.values()), type_hints.items()):
            for h in hint.__args__:
                _check_hint(h, value, param_name)
        return func(*args, **kwargs)

    return cast(F, wrapper)


class View(Sized, Reversible[T_co], ABC):
    """Abstract View class"""
    @abstractmethod
    def __init__(self, __x: Sequence[T_co]) -> None:
        self.__x = __x
        super().__init__()

    def __len__(self) -> int:
        return len(self.__x)

    def __reversed__(self) -> Iterator[T_co]:
        return reversed(self.__x)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.__x})'

    def __repr__(self) -> str:
        return self.__str__()


class PropView(View[T_co], Collection[T_co]):
    """View for DrawingProp"""
    def __init__(self, __props: Iterable[T_co], /) -> None:
        self.__props = list(__props)
        super().__init__(self.__props)

    def __contains__(self, __x: object) -> bool:
        return __x in self.__props

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.__props)


class CoordinatesView(View[Tuple[Nb, Nb]], Collection[Tuple[Nb, Nb]]):
    """View for coordinates"""
    def __init__(self, __coordinates: Iterable[Tuple[Nb, Nb]], /) -> None:
        self.__coordinates = list(__coordinates)
        super().__init__(self.__coordinates)

    def __contains__(self, __x: object) -> bool:
        return __x in self.__coordinates

    def __iter__(self) -> Iterator[Tuple[Nb, Nb]]:
        return iter(self.__coordinates)
