import pytest
import uarray as ua
import unumpy as np
import numpy as onp
import torch
import dask.array as da
import sparse
import unumpy.numpy_backend as NumpyBackend

import unumpy.torch_backend as TorchBackend
from unumpy.dask_backend import DaskBackend
import unumpy.sparse_backend as SparseBackend

ua.set_global_backend(NumpyBackend)

dtypes = ["int8", "int16", "int32", "float32", "float64"]
LIST_BACKENDS = [
    (NumpyBackend, (onp.ndarray, onp.generic)),
    (DaskBackend(), (da.Array, onp.generic)),
    (SparseBackend, (sparse.SparseArray, onp.ndarray, onp.generic)),
    pytest.param(
        (TorchBackend, (torch.Tensor,)),
        marks=pytest.mark.xfail(reason="PyTorch not fully NumPy compatible."),
    ),
]


FULLY_TESTED_BACKENDS = [NumpyBackend, DaskBackend]

try:
    import unumpy.xnd_backend as XndBackend
    import xnd
    from ndtypes import ndt

    LIST_BACKENDS.append((XndBackend, (xnd.xnd,)))
    FULLY_TESTED_BACKENDS.append(XndBackend)
except ImportError:
    XndBackend = None  # type: ignore
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
    (XndBackend, np.reciprocal),
    (XndBackend, np.isinf),
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
        (np.arange, (5, 20), {}),
        (np.arange, (5,), {}),
        (np.isinf, ([np.inf, np.NINF, 1.0, np.nan],), {}),
    ],
)
def test_ufuncs_coerce(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method(*args, **kwargs)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
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
        (np.ptp, ([1, 3, 2],), {}),
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
        # the following case tests the fix in Quansight-Labs/unumpy#36
        (np.compress, ([False, True], [[1, 2], [3, 4], [5, 6]], 1), {}),
        (np.extract, ([True, False, True, False], [0, 1, 2, 3]), {}),
        (np.count_nonzero, ([True, False, True, False],), {}),
        (np.linspace, (0, 100, 200), {}),
        (np.logspace, (0, 4, 200), {}),
        (np.diff, ([1, 3, 2],), {}),
        (np.isclose, ([1, 3, 2], [3, 2, 1]), {}),
        (np.allclose, ([1, 3, 2], [3, 2, 1]), {}),
        (np.isposinf, ([np.NINF, 0.0, np.inf],), {}),
        (np.isneginf, ([np.NINF, 0.0, np.inf],), {}),
        (np.iscomplex, ([1 + 1j, 1 + 0j, 4.5, 3, 2, 2j],), {}),
        (np.iscomplexobj, ([3, 1 + 0j],), {}),
        (np.isreal, ([1 + 1j, 1 + 0j, 4.5, 3, 2, 2j],), {}),
        (np.isrealobj, ([3, 1 + 0j],), {}),
        (np.isscalar, ([3.1],), {}),
        (np.array_equal, ([1, 2, 3], [1, 2, 3]), {}),
        (np.array_equiv, ([1, 2], [[1, 2], [1, 2]]), {}),
        (np.diag, ([1, 2, 3],), {}),
        (np.diagflat, ([[1, 2], [3, 4]],), {}),
        (np.copy, ([1, 2, 3],), {}),
        (np.tril, ([[1, 2], [3, 4]],), {}),
        (np.triu, ([[1, 2], [3, 4]],), {}),
        (np.vander, ([1, 2, 3, 5],), {}),
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
        assert isinstance(ret, tuple) and all(isinstance(s, int) for s in ret)
    elif method in (np.ndim, np.size):
        assert isinstance(ret, int)
    elif method in (
        np.allclose,
        np.iscomplex,
        np.iscomplexobj,
        np.isreal,
        np.isrealobj,
        np.isscalar,
        np.array_equal,
        np.array_equiv,
    ):
        assert isinstance(ret, (bool,) + types)
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
    if XndBackend is not None and backend == XndBackend:
        assert ret.dtype == ndt(dtype)
    else:
        assert ret.dtype == dtype


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.broadcast_arrays, ([1, 2], [[3, 4]]), {}),
        (np.meshgrid, ([1, 2, 3], [4, 5], [0, 1]), {}),
        (np.nonzero, ([3, 1, 2, 4],), {}),
        (np.where, ([[3, 1, 2, 4]],), {}),
        (np.gradient, ([[0, 1, 2], [3, 4, 5], [6, 7, 8]],), {}),
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
        (np.empty, (2,), {}),
        (np.empty_like, ([1, 2, 3],), {}),
        (np.eye, (2,), {}),
        (np.identity, (2,), {}),
        (np.full, ((1, 2, 3), 1.3), {}),
        (np.full_like, ([1, 2, 3], 2), {}),
        (np.ones, ((1, 2, 3),), {}),
        (np.ones_like, ([1, 2, 3],), {}),
        (np.zeros, ((1, 2, 3),), {}),
        (np.zeros_like, ([1, 2, 3],), {}),
        (np.asanyarray, ([1, 2, 3],), {}),
        (np.ascontiguousarray, ([1, 2, 3],), {}),
        (np.frombuffer, (), {}),
        (np.fromfunction, (lambda i: i + 1,), {"shape": (3,)}),
        (np.fromiter, (range(1, 4),), {}),
        (np.fromstring, ("1 2 3",), {"sep": " "}),
        (np.geomspace, (1, 1000), {"num": 4}),
        (np.tri, (3, 5, -1), {}),
    ],
)
def test_array_creation(backend, method, args, kwargs):
    backend, types = backend
    if method is np.frombuffer:
        buffer = onp.array([1, 2, 3]).tobytes()
        args = args + (buffer,)
    for dtype in dtypes:
        try:
            with ua.set_backend(backend, coerce=True):
                kwargs["dtype"] = dtype
                ret = method(*args, **kwargs)
        except ua.BackendNotImplementedError:
            if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
                raise
            pytest.xfail(reason="The backend has no implementation for this ufunc.")
        except TypeError:
            if method is np.asanyarray:
                raise pytest.xfail(
                    reason="The ufunc for this backend got an unexpected keyword."
                )
            else:
                raise

    assert isinstance(ret, types)

    if isinstance(ret, da.Array):
        ret.compute()
    if XndBackend is not None and backend == XndBackend:
        assert ret.dtype == ndt(dtype)
    else:
        assert ret.dtype == dtype


@pytest.mark.parametrize(
    "method, args, kwargs, res",
    [
        (np.add, ([1, 2], [3, 4]), {}, [4, 6]),
        (np.subtract, ([3, 4], [1, 2]), {}, [2, 2]),
        (np.multiply, ([1, 2], [4, 3]), {}, [4, 6]),
        (np.divide, ([6, 1], [3, 2]), {}, [2.0, 0.5]),
        (np.true_divide, ([6, 1], [3, 2]), {}, [2.0, 0.5]),
        (np.power, ([2, 3], [3, 2]), {}, [8, 9]),
        (np.positive, ([1, -2],), {}, [1, -2]),
        (np.negative, ([-2, 3],), {}, [2, -3]),
        (np.conj, ([1.0 + 2.0j, -1.0 - 1j],), {}, [1.0 - 2.0j, -1.0 + 1j]),
        (np.exp, ([0, 1, 2],), {}, [1.0, 2.718281828459045, 7.38905609893065]),
        (np.exp2, ([3, 4],), {}, [8, 16]),
        (np.log, ([1.0, np.e, np.e ** 2],), {}, [0.0, 1.0, 2.0]),
        (np.log2, ([1, 2, 2 ** 4],), {}, [0.0, 1.0, 4.0]),
        (np.log10, ([1e-5, -3.0],), {}, [-5.0, np.NaN]),
        (np.sqrt, ([1, 4, 9],), {}, [1, 2, 3]),
        (np.square, ([2, 3, 4],), {}, [4, 9, 16]),
        (np.cbrt, ([1, 8, 27],), {}, [1.0, 2.0, 3.0]),
        (np.reciprocal, ([1.0, 2.0, 4.0],), {}, [1.0, 0.5, 0.25]),
        (
            np.broadcast_to,
            ([1, 2, 3], (3, 3)),
            {},
            np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
        ),
    ],
)
def test_ufuncs_results(backend, method, args, kwargs, res):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method(*args, **kwargs)

            res = np.asarray(res)
            assert np.allclose(ret, res, equal_nan=True)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and backend is not XndBackend:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.linalg.multi_dot, ([[0, 1], [[1, 2], [3, 4]], [1, 0]],), {}),
        (np.linalg.matrix_power, ([[1, 2], [3, 4]], 2), {}),
        (np.linalg.cholesky, ([[1, -2j], [2j, 5]],), {}),
        (np.linalg.qr, ([[1, 2], [3, 4]],), {}),
        (np.linalg.svd, ([[1, 2], [3, 4]],), {}),
        (np.linalg.eig, ([[1, 1j], [-1j, 1]],), {}),
        (np.linalg.eigh, ([[1, -2j], [2j, 5]],), {}),
        (np.linalg.eigvals, ([[1, 2], [3, 4]],), {}),
        (np.linalg.eigvalsh, ([[1, -2j], [2j, 5]],), {}),
        (np.linalg.norm, ([[1, 2], [3, 4]],), {}),
        (np.linalg.cond, ([[1, 0, -1], [0, 1, 0], [1, 0, 1]],), {}),
        (np.linalg.det, ([[1, 2], [3, 4]],), {}),
        (np.linalg.matrix_rank, (np.eye(4),), {}),
        (np.linalg.slogdet, ([[1, 2], [3, 4]],), {}),
        (np.linalg.solve, ([[3, 1], [1, 2]], [9, 8]), {}),
        (
            np.linalg.tensorsolve,
            (
                np.eye((2 * 3 * 4)).reshape(2 * 3, 4, 2, 3, 4),
                np.empty(shape=(2 * 3, 4)),
            ),
            {},
        ),
        (np.linalg.lstsq, ([[3, 1], [1, 2]], [9, 8]), {"rcond": None}),
        (np.linalg.inv, ([[1.0, 2.0], [3.0, 4.0]],), {}),
        (np.linalg.pinv, ([[1.0, 2.0], [3.0, 4.0]],), {}),
        (np.linalg.tensorinv, (np.eye(4 * 6).reshape((4, 6, 8, 3)),), {}),
    ],
)
def test_linalg(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(NumpyBackend, coerce=True):
            ret = method(*args, **kwargs)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(ret, da.Array):
        ret.compute()
