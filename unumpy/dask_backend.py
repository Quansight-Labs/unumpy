import numpy as np
import dask.array as da
from uarray import Dispatchable, wrap_single_convertor, skip_backend
from unumpy import ufunc, ufunc_list, ndarray
import unumpy
import functools
import sys
import collections
import itertools
import random

from typing import Dict

_ufunc_mapping: Dict[ufunc, np.ufunc] = {}

__ua_domain__ = "numpy"


def wrap_map_blocks(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with skip_backend(sys.modules[__name__]):
            return da.map_blocks(func, *args, **kwargs)

    return wrapped


def wrap_uniform_create(func):
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
        with skip_backend(sys.modules[__name__]):
            for chunk_id in itertools.product(*map(lambda x: range(len(x)), chunks)):
                shape = tuple(chunks[i][j] for i, j in enumerate(chunk_id))
                dsk[(name,) + chunk_id] = func(shape, *args, **kwargs)

            dtype = str(func((), *args, **kwargs).dtype)

        return da.Array(dsk, name, chunks, dtype)

    return wrapped


_implementations: Dict = {
    unumpy.ufunc.__call__: wrap_map_blocks(unumpy.ufunc.__call__),
    unumpy.ones: wrap_uniform_create(unumpy.ones),
    unumpy.zeros: wrap_uniform_create(unumpy.zeros),
    unumpy.full: wrap_uniform_create(unumpy.full),
    unumpy.arange: lambda start, stop=None, step=None, **kw: da.arange(
        start, stop, step, **kw
    ),
}


def __ua_function__(method, args, kwargs):
    if method in _implementations:
        return _implementations[method](*args, **kwargs)

    if not hasattr(da, method.__name__):
        return NotImplemented

    return getattr(da, method.__name__)(*args, **kwargs)


@wrap_single_convertor
def __ua_convert__(value, dispatch_type, coerce):
    if dispatch_type is ndarray:
        if not coerce:
            return value
        return da.asarray(value) if value is not None else None

    return value


def replace_self(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        if self not in _ufunc_mapping:
            return NotImplemented

        return func(_ufunc_mapping[self], *args, **kwargs)

    return inner
