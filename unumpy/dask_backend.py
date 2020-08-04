import numpy as np
import dask.array as da
from uarray import (
    Dispatchable,
    wrap_single_convertor_instance,
    set_backend,
    get_state,
    set_state,
    set_backend,
)
from unumpy import ufunc, ufunc_list, ndarray
import unumpy
import functools
import sys
import collections
import itertools
import random

from typing import Dict


def overridden_class(self):
    if self is ndarray:
        return da.Array
    if self is ufunc:
        return da.ufunc.ufunc
    module = self.__module__.split(".")
    module = ".".join(m for m in module if m != "_multimethods")
    return _get_from_name_domain(self.__name__, module)


def _get_from_name_domain(name, domain):
    module = np
    domain_hierarchy = domain.split(".")
    for d in domain_hierarchy[1:]:
        module = getattr(module, d)
    if hasattr(module, name):
        return getattr(module, name)
    else:
        return NotImplemented


class DaskBackend:
    _ufunc_mapping: Dict[ufunc, np.ufunc] = {}
    __ua_domain__ = "numpy"

    def __init__(self, inner=None):
        from unumpy import numpy_backend as NumpyBackend

        _implementations: Dict = {
            unumpy.asarray: self.asarray,
            unumpy.ufunc.__call__: self.wrap_map_blocks(unumpy.ufunc.__call__),
            unumpy.ones: self.wrap_uniform_create(unumpy.ones),
            unumpy.zeros: self.wrap_uniform_create(unumpy.zeros),
            unumpy.full: self.wrap_uniform_create(unumpy.full),
            unumpy.trim_zeros: self.trim_zeros,
            unumpy.ClassOverrideMeta.overridden_class.fget: overridden_class,
        }

        self._implementations = _implementations
        self._inner = NumpyBackend if inner is None else inner

    @staticmethod
    def _wrap_current_state(func):
        state = get_state()

        @functools.wraps(func)
        def wrapped(*a, **kw):
            with set_state(state):
                return func(*a, **kw)

        return wrapped

    def asarray(self, a, dtype=None, order=None):
        if isinstance(a, da.Array):
            return self.wrap_map_blocks(unumpy.asarray)(a)
        with set_backend(self._inner, coerce=True):
            a = np.asarray(a, dtype=dtype, order=order)
        return da.from_array(a, asarray=False)

    def trim_zeros(self, filt, trim="fb"):
        nonzero_idxs = unumpy.nonzero(filt)[0].compute_chunk_sizes()

        if len(nonzero_idxs.compute()) == 0:
            return unumpy.asarray([], dtype=filt.dtype)

        start, end = None, None
        if "f" in trim:
            start = nonzero_idxs[0].compute()
        if "b" in trim:
            end = nonzero_idxs[-1].compute() + 1
        return filt[start:end]

    def wrap_map_blocks(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with set_backend(self._inner):
                return da.map_blocks(self._wrap_current_state(func), *args, **kwargs)

        return wrapped

    def wrap_uniform_create(self, func):
        @functools.wraps(func)
        def wrapped(shape, *args, **kwargs):
            if isinstance(shape, collections.abc.Iterable):
                shape = tuple(int(s) for s in shape)
            else:
                shape = (int(shape),)

            # Estimate 100 Mi elements per block
            blocksize = int((100 * (2 ** 20)) ** (1 / len(shape)))

            chunks = []
            for l in shape:
                chunks.append([])
                while l > 0:
                    s = max(min(blocksize, l), 0)
                    chunks[-1].append(s)
                    l -= s

            name = func.__name__ + "-" + hex(random.randrange(2 ** 64))
            dsk = {}
            with set_backend(self._inner):
                for chunk_id in itertools.product(
                    *map(lambda x: range(len(x)), chunks)
                ):
                    shape = tuple(chunks[i][j] for i, j in enumerate(chunk_id))
                    dsk[(name,) + chunk_id] = func(shape, *args, **kwargs)

                meta = func(tuple(0 for _ in shape), *args, **kwargs)
                dtype = str(meta.dtype)

            return da.Array(dsk, name, chunks, dtype=dtype, meta=meta)

        return wrapped

    def __ua_function__(self, method, args, kwargs):
        if method in self._implementations:
            return self._implementations[method](*args, **kwargs)

        if len(args) != 0 and isinstance(args[0], unumpy.ClassOverrideMeta):
            return NotImplemented

        if not hasattr(da, method.__name__):
            return NotImplemented

        return getattr(da, method.__name__)(*args, **kwargs)

    @wrap_single_convertor_instance
    def __ua_convert__(self, value, dispatch_type, coerce):
        if dispatch_type is not ufunc and value is None:
            return None

        if dispatch_type is ndarray:
            if not coerce and not isinstance(value, da.Array):
                return NotImplemented
            ret = da.asarray(value)
            with set_backend(self._inner):
                ret = ret.map_blocks(self._wrap_current_state(unumpy.asarray))

            return ret

        return value
