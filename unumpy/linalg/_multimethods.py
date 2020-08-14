import functools
import operator
from uarray import create_multimethod, mark_as, all_of_type, Dispatchable
import builtins

create_numpy = functools.partial(create_multimethod, domain="numpy.linalg")

from .._multimethods import (
    ndarray,
    _identity_argreplacer,
    _self_argreplacer,
    _dtype_argreplacer,
    mark_dtype,
    _first_argreplacer,
    _first2argreplacer,
    ndim,
)

__all__ = [
    "multi_dot",
    "matrix_power",
    "cholesky",
    "qr",
    "svd",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "norm",
    "cond",
    "det",
    "matrix_rank",
    "slogdet",
    "solve",
    "tensorsolve",
    "lstsq",
    "inv",
    "pinv",
    "tensorinv",
]


def multi_dot_default(arrays):
    res = arrays[0]
    for a in arrays[1:]:
        res = res @ a

    return res


@create_numpy(_first_argreplacer, default=multi_dot_default)
@all_of_type(ndarray)
def multi_dot(arrays):
    return arrays


def matrix_power_default(a, n):
    eigenvalues, eigenvectors = eig(a)
    diagonal = diag(eigenvalues)
    return multi_dot([eigenvectors, power(diagonal, n), inv(eigenvectors)])


@create_numpy(_self_argreplacer, default=matrix_power_default)
@all_of_type(ndarray)
def matrix_power(a, n):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def cholesky(a):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def qr(a, mode="reduced"):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def eig(a):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def eigh(a, UPLO="L"):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def eigvals(a):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def eigvalsh(a, UPLO="L"):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def norm(x, ord=None, axis=None, keepdims=False):
    return (x,)


def cond_default(x, p=None):
    if ndim(x) > 1:
        return norm_default(x, ord=p)
    else:
        raise ValueError("Array must be at least two-dimensional.")


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def cond(x, p=None):
    return (x,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def det(a):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def matrix_rank(M, tol=None, hermitian=False):
    return (M,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def slogdet(a):
    return (a,)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def solve(a, b):
    return (a, b)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def tensorsolve(a, b, axes=None):
    return (a, b)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def lstsq(a, b, rcond="warn"):
    return (a, b)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def inv(a):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def pinv(a, rcond=1e-15, hermitian=False):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def tensorinv(a, ind=2):
    return (a,)
