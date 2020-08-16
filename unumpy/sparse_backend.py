import numpy as np
import sparse
from uarray import Dispatchable, wrap_single_convertor
from unumpy import ufunc, ufunc_list, ndarray, dtype
from unumpy.random import RandomState
import unumpy
import functools

from typing import Dict

_ufunc_mapping: Dict[ufunc, np.ufunc] = {}

__ua_domain__ = "numpy"


def array(x, *args, **kwargs):
    if isinstance(x, sparse.SparseArray):
        return x

    if "dtype" in kwargs:
        dtype = kwargs["dtype"]
        return sparse.COO.from_numpy(np.asarray(x, dtype=dtype))

    return sparse.COO.from_numpy(np.asarray(x))


_class_mapping = {
    ndarray: sparse.SparseArray,
    dtype: np.dtype,
    ufunc: np.ufunc,
    RandomState: np.random.mtrand.RandomState,
}


def overridden_class(self):
    if self in _class_mapping:
        return _class_mapping[self]
    module = self.__module__.split(".")
    module = ".".join(m for m in module if m != "_multimethods")
    return _get_from_name_domain(self.__name__, module)


_implementations: Dict = {
    unumpy.ufunc.__call__: np.ufunc.__call__,
    unumpy.ufunc.reduce: np.ufunc.reduce,
    unumpy.array: array,
    unumpy.asarray: array,
    unumpy.ClassOverrideMeta.overridden_class.fget: overridden_class,
}


def _get_from_name_domain(name, domain):
    module = sparse
    domain_hierarchy = domain.split(".")
    for d in domain_hierarchy[1:]:
        if hasattr(module, d):
            module = getattr(module, d)
        else:
            return NotImplemented
    if hasattr(module, name):
        return getattr(module, name)
    else:
        return NotImplemented


def __ua_function__(method, args, kwargs):
    if method in _implementations:
        return _implementations[method](*args, **kwargs)

    if len(args) != 0 and isinstance(args[0], unumpy.ClassOverrideMeta):
        return NotImplemented

    sparse_method = _get_from_name_domain(method.__name__, method.domain)
    if sparse_method is NotImplemented:
        return NotImplemented

    return sparse_method(*args, **kwargs)


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
