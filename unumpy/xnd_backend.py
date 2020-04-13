try:
    import numpy as np
    from ndtypes import ndt
    import xnd
    import gumath.functions as fn
    import gumath as gu
    import uarray as ua
    from uarray import Dispatchable, wrap_single_convertor
    from unumpy import ufunc, ufunc_list, ndarray, dtype
    import unumpy
    import functools

    from typing import Dict

    _ufunc_mapping: Dict[ufunc, np.ufunc] = {}

    __ua_domain__ = "numpy"

    def gucall(self, *args, **kwargs):
        try:
            return self(*args, **kwargs)
        except (TypeError, ValueError):
            return NotImplemented

    _implementations: Dict = {
        unumpy.ufunc.__call__: gucall,
        unumpy.ufunc.reduce: gu.reduce,
    }

    def __ua_function__(method, args, kwargs):
        if method in _implementations:
            return _implementations[method](*args, **kwargs)

        return _generic(method, args, kwargs)

    @wrap_single_convertor
    def __ua_convert__(value, dispatch_type, coerce):
        if dispatch_type is ufunc and hasattr(fn, value.name):
            return getattr(fn, value.name)

        if value is None:
            return None

        if dispatch_type is ndarray:
            if not coerce and not isinstance(value, xnd.xnd):
                return NotImplemented

            return convert(value, coerce=coerce)

        if dispatch_type is dtype:
            return ndt(str(value))

        return NotImplemented

    def replace_self(func):
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            if self not in _ufunc_mapping:
                return NotImplemented

            return func(_ufunc_mapping[self], *args, **kwargs)

        return inner

    def _generic(method, args, kwargs):
        try:
            import numpy as np
            import unumpy.numpy_backend as NumpyBackend
        except ImportError:
            return NotImplemented

        with ua.set_backend(NumpyBackend, coerce=True):
            try:
                out = method(*args, **kwargs)
            except TypeError:
                return NotImplemented

        return convert_out(out, coerce=False)

    def convert_out(x, coerce):
        if isinstance(x, (tuple, list)):
            return type(x)(map(lambda x: convert_out(x, coerce=coerce), x))

        return convert(x, coerce=coerce)

    def convert(x, coerce):
        if isinstance(x, xnd.array):
            return x

        try:
            return xnd.array.from_buffer(memoryview(x))
        except TypeError:
            pass

        if coerce:
            return xnd.array(x)

        if isinstance(x, (int, float, bool)):
            return x

        raise ua.BackendNotImplementedError("Unsupported output received.")


except ImportError:
    pass
