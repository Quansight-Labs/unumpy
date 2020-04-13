import numpy as np
import sparse
from uarray import Dispatchable, wrap_single_convertor
from unumpy import ufunc, ufunc_list, ndarray
import unumpy
import functools

from typing import Dict

_ufunc_mapping: Dict[ufunc, np.ufunc] = {}

__ua_domain__ = "numpy"


def array(x, *args, **kwargs):
    if isinstance(x, sparse.SparseArray):
        return x

    return sparse.COO(np.asarray(x))


_implementations: Dict = {
    unumpy.ufunc.__call__: np.ufunc.__call__,
    unumpy.ufunc.reduce: np.ufunc.reduce,
    unumpy.array: array,
    unumpy.asarray: array,
}


def __ua_function__(method, args, kwargs):
    if method in _implementations:
        return _implementations[method](*args, **kwargs)

    if not hasattr(sparse, method.__name__):
        return NotImplemented

    return getattr(sparse, method.__name__)(*args, **kwargs)


@wrap_single_convertor
def __ua_convert__(value, dispatch_type, coerce):
    if dispatch_type is ufunc:
        return getattr(np, value.name)

    if value is None:
        return None

    if dispatch_type is ndarray:
        if not coerce:
            if not isinstance(value, sparse.SparseArray):
                return NotImplemented

        if isinstance(value, sparse.SparseArray):
            return value

        return sparse.as_coo(np.asarray(value))

    return value


def replace_self(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        if self not in _ufunc_mapping:
            return NotImplemented

        return func(_ufunc_mapping[self], *args, **kwargs)

    return inner
