import functools
import operator
from uarray import create_multimethod, mark_as, all_of_type, Dispatchable
import builtins

create_numpy = functools.partial(create_multimethod, domain="numpy.lib")

from .._multimethods import (
    ClassOverrideMetaWithGetAttr,
    _call_first_argreplacer,
    ndarray,
)


class ClassOverrideMetaForArrayterator(ClassOverrideMetaWithGetAttr):
    @create_numpy(
        _call_first_argreplacer,
        default=lambda self, var, buf_size=None: self.overridden_class(var, buf_size),
    )
    @all_of_type(ndarray)
    def __call__(self, var, buf_size=None):
        self._unwrapped = NotImplemented
        return (var,)


class Arrayterator(metaclass=ClassOverrideMetaForArrayterator):
    pass
