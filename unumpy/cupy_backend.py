try:
    import numpy as np
    import cupy as cp
    from uarray import Dispatchable, wrap_single_convertor
    from unumpy import ufunc, ufunc_list, ndarray
    import unumpy
    import functools

    from typing import Dict

    _ufunc_mapping: Dict[ufunc, np.ufunc] = {}

    __ua_domain__ = "numpy"

    _implementations: Dict = {}

    def _implements(np_func):
        def inner(func):
            _implementations[np_func] = func
            return func

        return inner

    def __ua_function__(method, args, kwargs):
        if method in _implementations:
            return _implementations[method](*args, **kwargs)

        if len(args) != 0 and isinstance(args[0], unumpy.ClassOverrideMeta):
            return NotImplemented

        if not hasattr(cp, method.__name__):
            return NotImplemented

        return getattr(cp, method.__name__)(*args, **kwargs)

    @wrap_single_convertor
    def __ua_convert__(value, dispatch_type, coerce):
        if dispatch_type is ufunc and hasattr(cp, value.name):
            return getattr(cp, value.name)

        if value is None:
            return None

        if dispatch_type is ndarray:
            if not coerce and not isinstance(value, cp.ndarray):
                return NotImplemented

            return cp.asarray(value)

        return value

    def replace_self(func):
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            if self not in _ufunc_mapping:
                return NotImplemented

            return func(_ufunc_mapping[self], *args, **kwargs)

        return inner

    @_implements(unumpy.ascontiguousarray)
    def _ascontiguousarray(arr, dtype=None):
        return cp.asarray(arr, dtype=dtype, order="C")

    @_implements(unumpy.asfortranarray)
    def _asfortranarray(arr, dtype=None):
        return cp.asarray(arr, dtype=dtype, order="F")

    @_implements(unumpy.ufunc.__call__)
    def _ufunc_call(self, *args, **kwargs):
        fname = self.name
        f = getattr(cp, fname, lambda *a, **kw: NotImplemented)
        return f(*args, **kwargs)


except ImportError:
    pass
