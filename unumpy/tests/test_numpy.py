import pytest
import uarray as ua
import unumpy as np
import numpy as onp
from ndtypes import ndt
import torch
import dask.array as da
import sparse
import unumpy.numpy_backend as NumpyBackend

import unumpy.torch_backend as TorchBackend
import unumpy.dask_backend as DaskBackend
import unumpy.sparse_backend as SparseBackend

ua.set_global_backend(NumpyBackend)

dtypes = ["int8", "int16", "int32", "float32", "float64"]
LIST_BACKENDS = [
    (NumpyBackend, (onp.ndarray, onp.generic)),
    (DaskBackend, (da.Array, onp.generic)),
    (SparseBackend, (sparse.SparseArray, onp.ndarray, onp.generic)),
    pytest.param(
        (TorchBackend, torch.Tensor),
        marks=pytest.mark.xfail(reason="PyTorch not fully NumPy compatible."),
    ),
]


FULLY_TESTED_BACKENDS = [NumpyBackend, DaskBackend]

try:
    import unumpy.xnd_backend as XndBackend
    import xnd

    LIST_BACKENDS.append((XndBackend, xnd.xnd))
    FULLY_TESTED_BACKENDS.append(XndBackend)
except ImportError:
    LIST_BACKENDS.append(
        pytest.param(
            (None, None), marks=pytest.mark.skip(reason="xnd is not importable")
        )
    )

try:
    import unumpy.cupy_backend as CupyBackend
    import cupy as cp

    LIST_BACKENDS.append(pytest.param((CupyBackend, (cp.ndarray, cp.generic))))
except ImportError:
    LIST_BACKENDS.append(
        pytest.param(
            (None, None), marks=pytest.mark.skip(reason="cupy is not importable")
        )
    )


EXCEPTIONS = {
    (DaskBackend, np.in1d),
    (DaskBackend, np.intersect1d),
    (DaskBackend, np.setdiff1d),
    (DaskBackend, np.setxor1d),
    (DaskBackend, np.union1d),
    (DaskBackend, np.sort),
    (DaskBackend, np.argsort),
    (DaskBackend, np.lexsort),
    (DaskBackend, np.partition),
    (DaskBackend, np.argpartition),
    (DaskBackend, np.sort_complex),
    (DaskBackend, np.msort),
    (DaskBackend, np.searchsorted),
}


@pytest.fixture(scope="session", params=LIST_BACKENDS)
def backend(request):
    backend = request.param
    return backend


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.add, ([1], [2]), {}),  # type: ignore
        (np.sin, ([1.0],), {}),  # type: ignore
        (np.arange, (5, 20, 5), {}),
    ],
)
def test_ufuncs_coerce(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method(*args, **kwargs)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    assert isinstance(ret, types)
    if isinstance(ret, da.Array):
        ret.compute()


def replace_args_kwargs(method, backend, args, kwargs):
    instance = ()
    while not hasattr(method, "_coerce_args"):
        instance += (method,)
        method = method.__call__

        if method is method.__call__:
            raise ValueError("Nowhere up the chain is there a multimethod.")

    args, kwargs, *_ = method._coerce_args(
        backend, instance + args, kwargs, coerce=True
    )
    return args[len(instance) :], kwargs


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.ndim, ([1, 2],), {}),
        (np.shape, ([1, 2],), {}),
        (np.size, ([1, 2],), {}),
        (np.any, ([True, False],), {}),
        (np.all, ([True, False],), {}),
        (np.min, ([1, 3, 2],), {}),
        (np.max, ([1, 3, 2],), {}),
        (np.argmin, ([1, 3, 2],), {}),
        (np.argmax, ([1, 3, 2],), {}),
        (np.nanargmin, ([1, 3, 2],), {}),
        (np.nanargmax, ([1, 3, 2],), {}),
        (np.nanmin, ([1, 3, 2],), {}),
        (np.nanmax, ([1, 3, 2],), {}),
        (np.unique, ([1, 2, 2],), {}),
        (np.in1d, ([1], [1, 2, 2]), {}),
        (np.isin, ([1], [1, 2, 2]), {}),
        (np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1]), {}),
        (np.setdiff1d, ([1, 3, 4, 3], [3, 1, 2, 1]), {}),
        (np.setxor1d, ([1, 3, 4, 3], [3, 1, 2, 1]), {}),
        (np.sort, ([3, 1, 2, 4],), {}),
        (np.lexsort, (([1, 2, 2, 3], [3, 1, 2, 1]),), {}),
        (np.stack, (([1, 2], [3, 4]),), {}),
        (np.concatenate, (([1, 2, 3], [3, 4]),), {}),
        (np.broadcast_to, ([1, 2], (2, 2)), {}),
        (np.argsort, ([3, 1, 2, 4],), {}),
        (np.msort, ([3, 1, 2, 4],), {}),
        (np.sort_complex, ([3.0 + 1.0j, 1.0 - 1.0j, 2.0 - 3.0j, 4 - 3.0j],), {}),
        (np.partition, ([3, 1, 2, 4], 2), {}),
        (np.argpartition, ([3, 1, 2, 4], 2), {}),
        (np.transpose, ([[3, 1, 2, 4]],), {}),
        (np.swapaxes, ([[1, 2, 3]], 0, 1), {}),
        (np.rollaxis, ([[1, 2, 3], [1, 2, 3]], 0, 1), {}),
        (np.moveaxis, ([[1, 2, 3], [1, 2, 3]], 0, 1), {}),
        (np.column_stack, ((((1, 2, 3)), ((1, 2, 3))),), {}),
        (np.hstack, ((((1, 2, 3)), ((1, 2, 3))),), {}),
        (np.vstack, ((((1, 2, 3)), ((1, 2, 3))),), {}),
        (np.block, ([([1, 2, 3]), ([1, 2, 3])],), {}),
        (np.reshape, ([[1, 2, 3], [1, 2, 3]], (6,)), {}),
        (np.argwhere, ([[3, 1, 2, 4]],), {}),
        (np.ravel, ([[3, 1, 2, 4]],), {}),
        (np.flatnonzero, ([[3, 1, 2, 4]],), {}),
        (np.where, ([[True, False, True, False]], [[1]], [[2]]), {}),
        (np.pad, ([1, 2, 3, 4, 5], (2, 3), "constant"), dict(constant_values=(4, 6))),
        (np.searchsorted, ([1, 2, 3, 4, 5], 2), {}),
        (np.compress, ([True, False, True, False], [0, 1, 2, 3]), {}),
        (np.extract, ([True, False, True, False], [0, 1, 2, 3]), {}),
        (np.count_nonzero, ([True, False, True, False],), {}),
    ],
)
def test_functions_coerce(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method(*args, **kwargs)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if method is np.shape:
        assert isinstance(ret, tuple)
    elif method in (np.ndim, np.size):
        assert isinstance(ret, int)
    else:
        assert isinstance(ret, types)

    if isinstance(ret, da.Array):
        ret.compute()


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.prod, ([1],), {}),
        (np.sum, ([1],), {}),
        (np.std, ([1, 3, 2],), {}),
        (np.var, ([1, 3, 2],), {}),
    ],
)
def test_functions_coerce_with_dtype(backend, method, args, kwargs):
    backend, types = backend
    for dtype in dtypes:
        try:
            with ua.set_backend(backend, coerce=True):
                kwargs["dtype"] = dtype
                ret = method(*args, **kwargs)
        except ua.BackendNotImplementedError:
            if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
                raise
            pytest.xfail(reason="The backend has no implementation for this ufunc.")

    assert isinstance(ret, types)
    if backend == XndBackend:
        assert ret.dtype == ndt(dtype)
    else:
        assert ret.dtype == dtype


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.broadcast_arrays, ([1, 2], [[3, 4]]), {}),
        (np.nonzero, ([3, 1, 2, 4],), {}),
        (np.where, ([[3, 1, 2, 4]],), {}),
    ],
)
def test_multiple_output(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method(*args, **kwargs)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    assert all(isinstance(arr, types) for arr in ret)

    for arr in ret:
        if isinstance(arr, da.Array):
            arr.compute()


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.eye, (2,), {}),
        (np.full, ((1, 2, 3), 1.3), {}),
        (np.ones, ((1, 2, 3),), {}),
        (np.zeros, ((1, 2, 3),), {}),
    ],
)
def test_array_creation(backend, method, args, kwargs):
    backend, types = backend
    for dtype in dtypes:
        try:
            with ua.set_backend(backend, coerce=True):
                kwargs["dtype"] = dtype
                ret = method(*args, **kwargs)
        except ua.BackendNotImplementedError:
            if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
                raise
            pytest.xfail(reason="The backend has no implementation for this ufunc.")

    assert isinstance(ret, types)

    if isinstance(ret, da.Array):
        ret.compute()
    if backend == XndBackend:
        assert ret.dtype == ndt(dtype)
    else:
        assert ret.dtype == dtype
