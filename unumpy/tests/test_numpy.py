import collections
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
    (
        NumpyBackend,
        (
            onp.ndarray,
            onp.generic,
            onp.ufunc,
            onp.random.RandomState,
            onp.random.Generator,
            onp.random.SeedSequence,
        ),
    ),
    (DaskBackend(), (da.Array, onp.generic, da.ufunc.ufunc, da.random.RandomState)),
    (
        SparseBackend,
        (sparse.SparseArray, onp.ndarray, onp.generic, onp.random.mtrand.RandomState),
    ),
    pytest.param(
        (TorchBackend, (torch.Tensor,)),
        marks=pytest.mark.xfail(reason="PyTorch not fully NumPy compatible."),
    ),
]


FULLY_TESTED_BACKENDS = [NumpyBackend, DaskBackend]

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
    CupyBackend = object()  # type: ignore


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
        (np.expand_dims, ([1, 2], 1), {}),
        (np.squeeze, ([[[0], [1], [2]]],), {}),
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
        (np.dstack, (([1, 2, 3], [2, 3, 4]),), {}),
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
        (np.unwrap, ([0.0, 0.78539816, 1.57079633, 5.49778714, 6.28318531],), {}),
        (np.around, ([0.5, 1.5, 2.5, 3.5, 4.5],), {}),
        (np.round_, ([0.5, 1.5, 2.5, 3.5, 4.5],), {}),
        (np.fix, ([2.1, 2.9, -2.1, -2.9],), {}),
        (np.cumprod, ([1, 2, 3],), {}),
        (np.cumsum, ([1, 2, 3],), {}),
        (np.nancumprod, ([1, np.nan],), {"axis": 0}),
        (np.nancumsum, ([1, np.nan],), {"axis": 0}),
        (np.diff, ([1, 3, 2],), {}),
        (np.ediff1d, ([1, 2, 4, 7, 0],), {}),
        (np.cross, ([1, 2, 3], [4, 5, 6]), {}),
        (np.trapz, ([1, 2, 3],), {}),
        (np.i0, ([0.0, 1.0],), {}),
        (np.sinc, ([0, 1, 2],), {}),
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
        (np.diagonal, ([[0, 1], [2, 3]],), {}),
        (np.diagflat, ([[1, 2], [3, 4]],), {}),
        (np.copy, ([1, 2, 3],), {}),
        (np.tril, ([[1, 2], [3, 4]],), {}),
        (np.triu, ([[1, 2], [3, 4]],), {}),
        (np.vander, ([1, 2, 3, 5],), {}),
        (np.tile, ([[1, 2], [3, 4]], 2), {}),
        (np.repeat, ([[1, 2], [3, 4]], 2), {"axis": 0}),
        (np.delete, ([1, 2, 3], 1), {}),
        (np.insert, ([1, 2, 3], 2, 0), {"axis": 0}),
        (np.append, ([1, 2, 3], [4, 5, 6]), {}),
        (np.resize, ([[1, 2], [3, 4]], (2, 3)), {}),
        (np.trim_zeros, ([0, 1, 2, 0],), {}),
        (np.flip, ([[1, 2], [3, 4]],), {"axis": 0}),
        (np.fliplr, ([[1, 2], [3, 4]],), {}),
        (np.flipud, ([[1, 2], [3, 4]],), {}),
        (np.roll, ([1, 2, 3], 1), {}),
        (np.rot90, ([[1, 2], [3, 4]],), {}),
        (np.angle, ([1.0, 1.0j, 1 + 1j],), {}),
        (np.real, ([1 + 2j, 3 + 4j, 5 + 6j],), {}),
        (np.imag, ([1 + 2j, 3 + 4j, 5 + 6j],), {}),
        (np.convolve, ([1, 2, 3], [0, 1, 0.5]), {}),
        (np.nan_to_num, ([np.inf, np.NINF, np.nan],), {}),
        (np.real_if_close, ([2.1 + 4e-14j, 5.2 + 3e-15j],), {}),
        (np.interp, (2.5, [1, 2, 3], [3, 2, 0]), {}),
        (np.indices, ((2, 3),), {}),
        (np.ravel_multi_index, ([[3, 6, 6], [4, 5, 1]], (7, 6)), {}),
        (np.take, ([4, 3, 5, 7, 6, 8], [0, 1, 4]), {}),
        (np.take_along_axis, ([[1, 2, 3], [4, 5, 6]], [[0, -1]], 1), {}),
        (np.choose, ([1, 2, 0], [[0, 1, 2], [10, 11, 12], [20, 21, 22]]), {}),
        (np.select, ([[True, False], [False, True]], [[0, 0], [1, 1]]), {}),
        (np.place, ([[1, 2], [3, 4]], [[False, False], [True, True]], [-99, 99]), {}),
        (np.put, ([0, 1, 2, 3, 4, 5], [0, 2], [-44, -55]), {}),
        (np.put_along_axis, ([[10, 30, 20], [60, 40, 50]], [[0], [1]], 99, 1), {}),
        (np.putmask, ([[1, 2], [3, 4]], [[False, False], [True, True]], [-99, 99]), {}),
        (np.fill_diagonal, ([[0, 0], [0, 0]], 1), {}),
        (np.nditer, ([[1, 2, 3]],), {}),
        (np.ndenumerate, ([[1, 2], [3, 4]],), {}),
        (np.ndindex, (3, 2, 1), {}),
        (np.lib.Arrayterator, ([[1, 2], [3, 4]],), {}),
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
    except TypeError:
        if backend is CupyBackend:
            if method is np.flip:
                pytest.xfail(reason="CuPy requires axis argument")
            elif method in {np.repeat, np.tile}:
                pytest.xfail(reason="CuPy does not accept array repeats")
        raise
    except ValueError:
        if isinstance(backend, DaskBackend) and method is np.place:
            pytest.xfail(reason="Default relies on delete and copyto")
        if backend is CupyBackend and method in {np.argwhere, np.block}:
            pytest.xfail(reason="Default relies on array_like coercion")
        raise
    except NotImplementedError:
        if backend is CupyBackend and method is np.sort_complex:
            pytest.xfail(reason="CuPy cannot sort complex data")
        raise
    except AttributeError:
        if backend is CupyBackend and method is np.lexsort:
            pytest.xfail(reason="CuPy doesn't accept tuples of arrays")
        raise

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
    elif method in {np.place, np.put, np.put_along_axis, np.putmask, np.fill_diagonal}:
        assert ret is None
    elif method in {np.nditer, np.ndenumerate, np.ndindex}:
        assert isinstance(ret, collections.abc.Iterator)
    elif method is np.lib.Arrayterator:
        assert isinstance(ret, collections.abc.Iterable)
    else:
        assert isinstance(ret, types)

    if isinstance(ret, da.Array):
        ret.compute()


def test_copyto(backend):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            dst = np.asarray([1, 2])
            src = np.asarray([3, 4])
            np.copyto(dst, src)
            assert np.array_equal(dst, src)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and (backend, np.copyto) not in EXCEPTIONS:
            raise pytest.xfail(
                reason="The backend has no implementation for this ufunc."
            )

    assert isinstance(dst, types)


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
        except TypeError:
            if backend is CupyBackend:
                if method in {np.std, np.var} and not dtype.startswith("float"):
                    pytest.xfail(reason="CuPy doesn't allow mean to cast to int")

    assert isinstance(ret, types)

    assert ret.dtype == dtype


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.broadcast_arrays, ([1, 2], [[3, 4]]), {}),
        (np.meshgrid, ([1, 2, 3], [4, 5], [0, 1]), {}),
        (np.nonzero, ([3, 1, 2, 4],), {}),
        (np.where, ([[3, 1, 2, 4]],), {}),
        (np.gradient, ([[0, 1, 2], [3, 4, 5], [6, 7, 8]],), {}),
        (np.split, ([1, 2, 3, 4], 2), {}),
        (np.array_split, ([1, 2, 3, 4, 5, 6, 7, 8], 3), {}),
        (np.dsplit, ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2), {}),
        (np.hsplit, ([[1, 2], [3, 4]], 2), {}),
        (np.vsplit, ([[1, 2], [3, 4]], 2), {}),
        (np.ix_, ([0, 1], [2, 4]), {}),
        (np.unravel_index, ([22, 41, 37], (7, 6)), {}),
        (np.diag_indices, (4,), {}),
        (np.diag_indices_from, ([[1, 2], [3, 4]],), {}),
        (np.mask_indices, (3, np.triu), {}),
        (np.tril_indices, (4,), {}),
        (np.tril_indices_from, ([[1, 2], [3, 4]],), {}),
        (np.triu_indices, (4,), {}),
        (np.triu_indices_from, ([[1, 2], [3, 4]],), {}),
        (np.nested_iters, ([[0, 1], [2, 3]], [[0], [1]]), {}),
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
    except ValueError:
        if backend is SparseBackend:
            if method in {
                np.mask_indices,
                np.tril_indices,
                np.tril_indices_from,
                np.triu_indices,
                np.triu_indices_from,
            }:
                raise pytest.xfail(
                    reason="Sparse's methods for triangular matrices require an array with zero fill-values as argument."
                )
    except TypeError:
        if backend is CupyBackend and method is np.unravel_index:
            pytest.xfail(reason="cupy.unravel_index is broken in version 6.0")
        raise
    if method is np.nested_iters:
        assert all(isinstance(ite, collections.abc.Iterator) for ite in ret)
    else:
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
        (np.asfarray, ([1, 2, 3],), {}),
        (np.asfortranarray, ([[1, 2], [3, 4]],), {}),
        (np.asarray_chkfinite, ([1, 2, 3],), {}),
        (np.require, ([[1, 2], [3, 4]],), {}),
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
        (np.float_power, ([2, 3], [3, 2]), {}, [8, 9]),
        (np.positive, ([1, -2],), {}, [1, -2]),
        (np.negative, ([-2, 3],), {}, [2, -3]),
        (np.conjugate, ([1.0 + 2.0j, -1.0 - 1j],), {}, [1.0 - 2.0j, -1.0 + 1j]),
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
        (
            np.degrees,
            ([0.0, 0.52359878, 1.04719755, 1.57079633],),
            {},
            [0.0, 30.0, 60.0, 90.0],
        ),
        (
            np.radians,
            ([0.0, 30.0, 60.0, 90.0],),
            {},
            [0.0, 0.52359878, 1.04719755, 1.57079633],
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
        if backend in FULLY_TESTED_BACKENDS:
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
        (np.linalg.lstsq, ([[3, 1], [1, 2]], [9, 8]), {}),
        (np.linalg.inv, ([[1.0, 2.0], [3.0, 4.0]],), {}),
        (np.linalg.pinv, ([[1.0, 2.0], [3.0, 4.0]],), {}),
        (np.linalg.tensorinv, (np.eye(4 * 6).reshape((4, 6, 8, 3)),), {}),
    ],
)
def test_linalg(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method(*args, **kwargs)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if method in {
        np.linalg.qr,
        np.linalg.svd,
        np.linalg.eig,
        np.linalg.eigh,
        np.linalg.slogdet,
        np.linalg.lstsq,
    }:
        assert all(isinstance(arr, types) for arr in ret)

        for arr in ret:
            if isinstance(arr, da.Array):
                arr.compute()
    else:
        assert isinstance(ret, types)

        if isinstance(ret, da.Array):
            ret.compute()


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.random.default_rng, (42,), {}),
        (np.random.RandomState, (42,), {}),
        # (np.random.Generator, (), {}),
        # (np.random.BitGenerator, (), {}),
        (np.random.SeedSequence, (42,), {}),
        (np.random.rand, (1, 2), {}),
        (np.random.randn, (1, 2), {}),
        (np.random.randint, ([1, 2],), {}),
        (np.random.random_integers, (2,), {}),
        (np.random.random_sample, (), {"size": 2}),
        (np.random.random, (), {"size": 2}),
        (np.random.ranf, (), {"size": 2}),
        (np.random.sample, (), {"size": 2}),
        (np.random.choice, ([1, 2],), {}),
        (np.random.bytes, (10,), {}),
        (np.random.shuffle, ([1, 2, 3, 4],), {}),
        (np.random.permutation, ([1, 2, 3, 4],), {}),
        (np.random.beta, (1, 2), {"size": 2}),
        (np.random.binomial, (10, 0.5), {"size": 2}),
        (np.random.chisquare, (2,), {"size": 2}),
        (np.random.dirichlet, ((10, 5, 3),), {}),
        (np.random.exponential, (), {"size": 2}),
        (np.random.f, (1.0, 48.0), {"size": 2}),
        (np.random.gamma, (2.0, 2.0), {"size": 2}),
        (np.random.geometric, (0.35,), {"size": 2}),
        (np.random.gumbel, (0.0, 0.1), {"size": 2}),
        (np.random.hypergeometric, (100, 2, 10), {"size": 2}),
        (np.random.laplace, (0.0, 1.0), {"size": 2}),
        (np.random.logistic, (10, 1), {"size": 2}),
        (np.random.lognormal, (3.0, 1.0), {"size": 2}),
        (np.random.logseries, (0.6,), {"size": 2}),
        (np.random.multinomial, (20, [1 / 6.0] * 6), {}),
        (np.random.multivariate_normal, ([0, 0], [[1, 0], [0, 100]]), {}),
        (np.random.negative_binomial, (1, 0.1), {"size": 2}),
        (np.random.noncentral_chisquare, (3, 20), {"size": 2}),
        (np.random.noncentral_f, (3, 20, 3.0), {"size": 2}),
        (np.random.normal, (0, 0.1), {"size": 2}),
        (np.random.pareto, (3.0,), {"size": 2}),
        (np.random.poisson, (5,), {"size": 2}),
        (np.random.power, (5.0,), {"size": 2}),
        (np.random.rayleigh, (3,), {"size": 2}),
        (np.random.standard_cauchy, (), {"size": 2}),
        (np.random.standard_exponential, (), {"size": 2}),
        (np.random.standard_gamma, (2.0,), {"size": 2}),
        (np.random.standard_normal, (), {"size": 2}),
        (np.random.standard_t, (10,), {"size": 2}),
        (np.random.triangular, (-3, 0, 8), {"size": 2}),
        (np.random.uniform, (-1, 0), {"size": 2}),
        (np.random.vonmises, (0.0, 4.0), {"size": 2}),
        (np.random.wald, (3, 2), {"size": 2}),
        (np.random.weibull, (5.0,), {"size": 2}),
        (np.random.zipf, (2.0,), {"size": 2}),
        (np.random.seed, (), {}),
        (np.random.get_state, (), {}),
        # (np.random.set_state, (), {}),
    ],
)
def test_random(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method(*args, **kwargs)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if method is np.random.bytes:
        assert isinstance(ret, bytes)
    elif method in {np.random.shuffle, np.random.seed}:
        assert ret is None
    elif method is np.random.get_state:
        assert isinstance(ret, tuple)
    else:
        assert isinstance(ret, types)

    if isinstance(ret, da.Array):
        ret.compute()


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (np.random.Generator.random, (), {"size": 2}),
        (np.random.Generator.choice, ([1, 2],), {}),
        (np.random.Generator.bytes, (10,), {}),
        (np.random.Generator.shuffle, ([1, 2, 3, 4],), {}),
        (np.random.Generator.permutation, ([1, 2, 3, 4],), {}),
        (np.random.Generator.beta, (1, 2), {"size": 2}),
        (np.random.Generator.binomial, (10, 0.5), {"size": 2}),
        (np.random.Generator.chisquare, (2,), {"size": 2}),
        (np.random.Generator.dirichlet, ((10, 5, 3),), {}),
        (np.random.Generator.exponential, (), {"size": 2}),
        (np.random.Generator.f, (1.0, 48.0), {"size": 2}),
        (np.random.Generator.gamma, (2.0, 2.0), {"size": 2}),
        (np.random.Generator.geometric, (0.35,), {"size": 2}),
        (np.random.Generator.gumbel, (0.0, 0.1), {"size": 2}),
        (np.random.Generator.hypergeometric, (100, 2, 10), {"size": 2}),
        (np.random.Generator.laplace, (0.0, 1.0), {"size": 2}),
        (np.random.Generator.logistic, (10, 1), {"size": 2}),
        (np.random.Generator.lognormal, (3.0, 1.0), {"size": 2}),
        (np.random.Generator.logseries, (0.6,), {"size": 2}),
        (np.random.Generator.multinomial, (20, [1 / 6.0] * 6), {}),
        (np.random.Generator.multivariate_normal, ([0, 0], [[1, 0], [0, 100]]), {}),
        (np.random.Generator.negative_binomial, (1, 0.1), {"size": 2}),
        (np.random.Generator.noncentral_chisquare, (3, 20), {"size": 2}),
        (np.random.Generator.noncentral_f, (3, 20, 3.0), {"size": 2}),
        (np.random.Generator.normal, (0, 0.1), {"size": 2}),
        (np.random.Generator.pareto, (3.0,), {"size": 2}),
        (np.random.Generator.poisson, (5,), {"size": 2}),
        (np.random.Generator.power, (5.0,), {"size": 2}),
        (np.random.Generator.rayleigh, (3,), {"size": 2}),
        (np.random.Generator.standard_cauchy, (), {"size": 2}),
        (np.random.Generator.standard_exponential, (), {"size": 2}),
        (np.random.Generator.standard_gamma, (2.0,), {"size": 2}),
        (np.random.Generator.standard_normal, (), {"size": 2}),
        (np.random.Generator.standard_t, (10,), {"size": 2}),
        (np.random.Generator.triangular, (-3, 0, 8), {"size": 2}),
        (np.random.Generator.uniform, (-1, 0), {"size": 2}),
        (np.random.Generator.vonmises, (0.0, 4.0), {"size": 2}),
        (np.random.Generator.wald, (3, 2), {"size": 2}),
        (np.random.Generator.weibull, (5.0,), {"size": 2}),
        (np.random.Generator.zipf, (2.0,), {"size": 2}),
    ],
)
def test_Generator(backend, method, args, kwargs):
    backend, types = backend
    try:
        with ua.set_backend(backend, coerce=True):
            rng = np.random.default_rng()
            ret = method(rng, *args, **kwargs)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if method is np.random.Generator.bytes:
        assert isinstance(ret, bytes)
    elif method is np.random.Generator.shuffle:
        assert ret is None
    else:
        assert isinstance(ret, types)

    if isinstance(ret, da.Array):
        ret.compute()


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        (
            np.apply_along_axis,
            (lambda a: (a[0] + a[-1]) * 0.5, 1, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            {},
        ),
        (np.apply_over_axes, (np.sum, [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [0, 1]), {}),
        (np.frompyfunc, (bin, 1, 1), {}),
        (
            np.piecewise,
            (
                [0, 1, 2, 3],
                [[True, False, True, False], [False, True, False, True]],
                [0, 1],
            ),
            {},
        ),
    ],
)
def test_functional(backend, method, args, kwargs):
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


@pytest.mark.parametrize(
    "method, args",
    [
        (np.c_, ([1, 2, 3], [4, 5, 6])),
        (np.r_, ([1, 2, 3], [4, 5, 6])),
        (np.s_, (slice(2, None, 2),)),
    ],
)
def test_class_getitem(backend, method, args):
    backend, types, = backend
    try:
        with ua.set_backend(backend, coerce=True):
            ret = method[args]
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS and (backend, method) not in EXCEPTIONS:
            raise
        pytest.xfail(reason="The backend has no implementation for this class.")

    if method is np.s_:
        all(isinstance(s, slice) for s in ret)
    else:
        assert isinstance(ret, types)


def test_class_overriding():
    with ua.set_backend(NumpyBackend, coerce=True):
        assert isinstance(onp.add, np.ufunc)
        assert isinstance(onp.dtype("float64"), np.dtype)
        assert np.dtype("float64") == onp.float64
        assert isinstance(np.dtype("float64"), onp.dtype)
        assert isinstance(onp.random.RandomState(), np.random.RandomState)
        assert isinstance(onp.random.Generator(onp.random.PCG64()), np.random.Generator)
        assert issubclass(onp.ufunc, np.ufunc)
        assert issubclass(onp.random.RandomState, np.random.RandomState)
        assert issubclass(onp.random.Generator, np.random.Generator)

    with ua.set_backend(DaskBackend(), coerce=True):
        assert isinstance(da.add, np.ufunc)
        assert isinstance(onp.dtype("float64"), np.dtype)
        assert np.dtype("float64") == onp.float64
        assert isinstance(np.dtype("float64"), onp.dtype)
        assert isinstance(da.random.RandomState(), np.random.RandomState)
        assert issubclass(da.ufunc.ufunc, np.ufunc)
        assert issubclass(da.random.RandomState, np.random.RandomState)

    with ua.set_backend(SparseBackend, coerce=True):
        assert isinstance(onp.add, np.ufunc)
        assert isinstance(onp.dtype("float64"), np.dtype)
        assert np.dtype("float64") == onp.float64
        assert isinstance(np.dtype("float64"), onp.dtype)
        assert isinstance(onp.random.RandomState(), np.random.RandomState)
        assert isinstance(onp.random.Generator(onp.random.PCG64()), np.random.Generator)
        assert issubclass(onp.ufunc, np.ufunc)
        assert issubclass(onp.random.RandomState, np.random.RandomState)
        assert issubclass(onp.random.Generator, np.random.Generator)

    if hasattr(CupyBackend, "__ua_function__"):
        with ua.set_backend(CupyBackend, coerce=True):
            assert isinstance(cp.add, np.ufunc)
            assert isinstance(cp.dtype("float64"), np.dtype)
            assert np.dtype("float64") == cp.float64
            assert isinstance(np.dtype("float64"), cp.dtype)
            assert isinstance(cp.random.RandomState(), np.random.RandomState)
            assert issubclass(cp.ufunc, np.ufunc)
            assert issubclass(cp.random.RandomState, np.random.RandomState)
