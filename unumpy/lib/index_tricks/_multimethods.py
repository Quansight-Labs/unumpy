import functools
import operator
from uarray import create_multimethod, mark_as, all_of_type, Dispatchable
import builtins

create_numpy = functools.partial(create_multimethod, domain="numpy.lib.index_tricks")

from ..._multimethods import ClassOverrideMetaWithConstructorAndGetAttr


class CClass(metaclass=ClassOverrideMetaWithConstructorAndGetAttr):
    def __getitem__(self, key):
        return CClass().__getitem__(key)


c_ = CClass()


class RClass(metaclass=ClassOverrideMetaWithConstructorAndGetAttr):
    def __getitem__(self, key):
        return RClass().__getitem__(key)


r_ = RClass()


class IndexExpression(metaclass=ClassOverrideMetaWithConstructorAndGetAttr):
    def __init__(self, maketuple):
        self.maketuple = maketuple

    def __getitem__(self, key):
        return IndexExpression(self.maketuple).__getitem__(key)


s_ = IndexExpression(maketuple=True)
