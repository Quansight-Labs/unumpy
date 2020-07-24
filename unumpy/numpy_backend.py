import numpy as np
from uarray import Dispatchable, wrap_single_convertor
from unumpy import ufunc, ufunc_list, ndarray, dtype, linalg
import unumpy
import functools

from typing import Dict

_ufunc_mapping: Dict[ufunc, np.ufunc] = {}

__ua_domain__ = "numpy"


def overridden_class(self):
    module = self.__module__.split(".")
    module = ".".join(m for m in module if m != "_multimethods")
    return _get_from_name_domain(self.__name__, module)


_implementations: Dict = {
    unumpy.ufunc.__call__: np.ufunc.__call__,
    unumpy.ufunc.reduce: np.ufunc.reduce,
    unumpy.count_nonzero: lambda a, axis=None: np.asarray(np.count_nonzero(a, axis))[
        ()
    ],
    unumpy.ClassOverrideMeta.overridden_class.fget: overridden_class,
}


def _get_from_name_domain(name, domain):
    module = np
    domain_hierarchy = domain.split(".")
    for d in domain_hierarchy[1:]:
        module = getattr(module, d)
    if hasattr(module, name):
        return getattr(module, name)
    else:
        return NotImplemented


def __ua_function__(method, args, kwargs):
    if method in _implementations:
        return _implementations[method](*args, **kwargs)

    method_numpy = _get_from_name_domain(method.__name__, method.domain)
    if method_numpy is NotImplemented:
        return NotImplemented

    return method_numpy(*args, **kwargs)


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
