"""Internal types module"""
from __future__ import annotations

import sys
from abc import ABC, ABCMeta, abstractmethod
from functools import wraps
from os import PathLike
from typing import (
    Any, Callable, ChainMap, Collection, Dict, Generic, Iterable, Iterator, Reversible, Sequence,
    Tuple, TypeVar, Union, cast, get_args, get_origin, overload
)

from numpy.typing import NDArray
from typing_extensions import Annotated, get_type_hints

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
if sys.version_info >= (3, 9):
    AnyPath = Union[PathLike[str], str]
else:
    AnyPath = Union[PathLike, str]
SomeArrayLike = Union[Sequence[float], NDArray[Any]]


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


class View(Reversible[T], Collection[T]):
    """Abstract View class"""
    __slots__ = '__x'

    def __init__(self, __x: Collection[T]) -> None:
        self.__x = __x
        super().__init__()

    def __contains__(self, __x: object) -> bool:
        return __x in self.__x

    def __iter__(self) -> Iterator[T]:
        return iter(self.__x)

    def __len__(self) -> int:
        return len(self.__x)

    def __reversed__(self) -> Iterator[T]:
        return reversed(tuple(self.__x))

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.__x})'

    __repr__ = __str__


class NamedMutableSequenceMeta(ABCMeta):
    __slots__: Tuple[str, ...] = ()

    def __new__(cls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any],
                ignore_slots: bool = False) -> NamedMutableSequenceMeta:
        # Let's use __slots__ only if the class is a concrete application
        if ignore_slots:
            return super().__new__(cls, name, bases, namespace)

        # dict.fromkeys works as an OrderedSet
        abases = dict.fromkeys(b for base in bases for b in base.__mro__)
        # Remove useless base classes
        # We could only yeet object but Python 3.8 doesn't have annotated abc in typing
        for clsb in NamedMutableSequence.__mro__:
            del abases[clsb]
        # Get annotations in reverse mro order for the variable names
        # We don't need to reverse abases because of the behaviour of ChainMap
        types = ChainMap(*(base.__annotations__ for base in abases))
        types.update(namespace.get('__annotations__', {}))
        # Finally add the variable names
        namespace['__slots__'] = tuple(types.keys())
        return super().__new__(cls, name, bases, namespace)


class NamedMutableSequence(Sequence[T_co], ABC, ignore_slots=True, metaclass=NamedMutableSequenceMeta):
    __slots__: Tuple[str, ...] = ()
    __annotations__: Dict[str, Any] = {}
    __annotations__.clear()

    def __init__(self, *args: T_co, **kwargs: T_co) -> None:
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        if args:
            for k, v in zip(self.__slots__, args):
                self.__setattr__(k, v)

    def __str__(self) -> str:
        clsname = self.__class__.__name__
        values = ', '.join('%s=%r' % (k, self.__getattribute__(k)) for k in self.__slots__)
        return '%s(%s)' % (clsname, values)

    __repr__ = __str__

    @overload
    def __getitem__(self, index: int) -> T_co:
        ...

    @overload
    def __getitem__(self, index: slice) -> Tuple[T_co, ...]:
        ...

    def __getitem__(self, index: int | slice) -> T_co | Tuple[T_co, ...]:
        if isinstance(index, slice):
            return tuple(
                self.__getattribute__(self.__slots__[i])
                for i in range(index.start, index.stop)
            )
        return self.__getattribute__(self.__slots__[index])

    def __setitem__(self, item: int, value: Any) -> None:
        self.__setattr__(self.__slots__[item], value)

    def __len__(self) -> int:
        return self.__slots__.__len__()
