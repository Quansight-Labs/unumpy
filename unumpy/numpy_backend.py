import numpy as np
from uarray import Dispatchable, wrap_single_convertor
from unumpy import ufunc, ufunc_list, ndarray, dtype
import unumpy
import functools

from typing import Dict

_ufunc_mapping: Dict[ufunc, np.ufunc] = {}

__ua_domain__ = "numpy"


_implementations: Dict = {
    unumpy.ufunc.__call__: np.ufunc.__call__,
    unumpy.ufunc.reduce: np.ufunc.reduce,
    unumpy.count_nonzero: lambda a, axis=None: np.asarray(np.count_nonzero(a, axis))[
        ()
    ],
}


def __ua_function__(method, args, kwargs):
    if method in _implementations:
        return _implementations[method](*args, **kwargs)

    if not hasattr(np, method.__name__):
        return NotImplemented

    return getattr(np, method.__name__)(*args, **kwargs)


@wrap_single_convertor
def __ua_convert__(value, dispatch_type, coerce):
    if dispatch_type is ufunc:
        return getattr(np, value.name)

    if value is None:
        return None

    if dispatch_type is ndarray:
        if not coerce and not isinstance(value, np.ndarray):
            return NotImplemented

        return np.asarray(value)

    if dispatch_type is dtype:
        try:
            return np.dtype(str(value))
        except TypeError:
            return np.dtype(value)

    return value


def replace_self(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        if self not in _ufunc_mapping:
            return NotImplemented

        return func(_ufunc_mapping[self], *args, **kwargs)

    return inner
