import functools
import operator
from uarray import create_multimethod, mark_as, all_of_type, Dispatchable
import builtins

create_numpy = functools.partial(create_multimethod, domain="numpy.lib")

from .._multimethods import ClassOverrideMetaWithConstructorAndGetAttr


class Arrayterator(metaclass=ClassOverrideMetaWithConstructorAndGetAttr):
    pass
