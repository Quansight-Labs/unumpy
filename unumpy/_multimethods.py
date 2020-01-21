import functools
import operator
from uarray import create_multimethod, mark_as, all_of_type, Dispatchable
import builtins

create_numpy = functools.partial(create_multimethod, domain="numpy")


def _identity_argreplacer(args, kwargs, arrays):
    return args, kwargs


def _dtype_argreplacer(args, kwargs, dispatchables):
    def replacer(*a, dtype=None, **kw):
        out_kw = kw.copy()
        out_kw["dtype"] = dispatchables[0]
        return a, out_kw

    return replacer(*args, **kwargs)


def _self_argreplacer(args, kwargs, dispatchables):
    def self_method(a, *args, **kwargs):
        return dispatchables + args, kwargs

    return self_method(*args, **kwargs)


def _ureduce_argreplacer(args, kwargs, dispatchables):
    def ureduce(self, a, axis=0, dtype=None, out=None, keepdims=False):
        return (
            (dispatchables[0], dispatchables[1]),
            dict(
                axis=axis,
                dtype=dispatchables[2],
                out=dispatchables[3],
                keepdims=keepdims,
            ),
        )

    return ureduce(*args, **kwargs)


def _reduce_argreplacer(args, kwargs, arrays):
    def reduce(a, axis=None, dtype=None, out=None, keepdims=False):
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype

        if keepdims is not False:
            kwargs["keepdims"] = keepdims

        return ((arrays[0],), dict(axis=axis, out=arrays[1], **kwargs))

    return reduce(*args, **kwargs)


def _first2argreplacer(args, kwargs, arrays):
    def func(a, b, **kw):
        kw_out = kw.copy()
        if "out" in kw:
            kw_out["out"] = arrays[2]
        return arrays[:2], kw_out

    return func(*args, **kwargs)


def getattr_impl(attr):
    def func(a):
        if hasattr(a, attr):
            return getattr(a, attr)

        return NotImplemented

    return func


def method_impl(method):
    def func(self, *a, **kw):
        if hasattr(a, method):
            return getattr(a, method)(*a, **kw)

        return NotImplemented


def _ufunc_argreplacer(args, kwargs, arrays):
    self = args[0]
    args = args[1:]
    in_arrays = arrays[1 : self.nin + 1]
    out_arrays = arrays[self.nin + 1 : -1]
    dtype = arrays[-1]
    if self.nout == 1:
        out_arrays = out_arrays[0]

    if "out" in kwargs:
        kwargs = {**kwargs, "out": out_arrays}
    if "dtype" in kwargs:
        kwargs["dtype"] = dtype

    return (arrays[0], *in_arrays), kwargs


def _math_op(name, inplace=True, reverse=True):
    def f(self, other):
        return globals()[name](self, other)

    def r(self, other):
        return globals()[name](other, self)

    def i(self, other):
        return globals()[name](self, other, out=self)

    out = [f]

    if reverse:
        out.append(r)

    if inplace:
        out.append(i)

    return out if len(out) != 1 else out[0]


def _unary_op(name):
    def f(self):
        return globals()[name](self)

    return f


class ndarray:
    __add__, __radd__, __iadd__ = _math_op("add")
    __sub__, __rsub__, __isub__ = _math_op("subtract")
    __mul__, __rmul__, __imul__ = _math_op("multiply")
    __truediv__, __rtruediv__, __itruediv__ = _math_op("true_divide")
    __floordiv__, __rfloordiv__, __ifloordiv__ = _math_op("floor_divide")
    __matmul__, __rmatmul__, __imatmul__ = _math_op("matmul")
    __mod__, __rmod__, __imod__ = _math_op("mod")
    __divmod__, __rdivmod__ = _math_op("divmod", reverse=False)
    __lshift__, __rlshift__, __ilshift__ = _math_op("left_shift")
    __rshift__, __rrshift__, __irshift__ = _math_op("right_shift")
    __pow__, __rpow__, __ipow__ = _math_op("power")
    __and__, __rand__, __iand__ = _math_op("bitwise_and")
    __or__, __ror__, __ior__ = _math_op("bitwise_or")
    __xor__, __rxor__, __ixor__ = _math_op("bitwise_xor")
    __neg__ = _unary_op("negative")
    __pos__ = _unary_op("positive")
    __abs__ = _unary_op("absolute")
    __invert__ = _unary_op("invert")
    __lt__ = _math_op("less", inplace=False, reverse=False)
    __gt__ = _math_op("greater", inplace=False, reverse=False)
    __le__ = _math_op("less_equal", inplace=False, reverse=False)
    __ge__ = _math_op("greater_equal", inplace=False, reverse=False)
    __eq__ = _math_op("equal", inplace=False, reverse=False)
    __ne__ = _math_op("not_equal", inplace=False, reverse=False)

    def __array_ufunc__(self, method, *inputs, **kwargs):
        return NotImplemented


class dtype:
    pass


class ufunc:
    def __init__(self, name, nin, nout):
        self.name = name
        self.nin, self.nout = nin, nout

    def __str__(self):
        return "<ufunc '{}'>".format(self.name)

    __repr__ = __str__

    @property  # type: ignore
    @create_numpy(_self_argreplacer)
    def types(self):
        return (mark_ufunc(self),)

    @property  # type: ignore
    @create_numpy(_self_argreplacer)
    def identity(self):
        return (mark_ufunc(self),)

    @property
    def nargs(self):
        return self.nin + self.nout

    @property
    def ntypes(self):
        return len(self.types)

    @create_numpy(_ufunc_argreplacer)
    @all_of_type(ndarray)
    def __call__(self, *args, out=None, dtype=None):
        in_args = tuple(args)
        dtype = mark_dtype(dtype)
        if not isinstance(out, tuple):
            out = (out,)

        return (
            (mark_ufunc(self),)
            + in_args
            + tuple(mark_non_coercible(o) for o in out)
            + (dtype,)
        )

    @create_numpy(_ureduce_argreplacer)
    @all_of_type(ndarray)
    def reduce(self, a, axis=0, dtype=None, out=None, keepdims=False):
        return (mark_ufunc(self), a, mark_dtype(dtype), mark_non_coercible(out))

    @create_numpy(_ureduce_argreplacer)
    @all_of_type(ndarray)
    def accumulate(self, a, axis=0, dtype=None, out=None):
        return (mark_ufunc(self), a, mark_dtype(dtype), mark_non_coercible(out))


mark_ufunc = mark_as(ufunc)
mark_dtype = mark_as(dtype)
mark_non_coercible = lambda x: Dispatchable(x, ndarray, coercible=False)

# Math operations
add = ufunc("add", 2, 1)
subtract = ufunc("subtract", 2, 1)
multiply = ufunc("multiply", 2, 1)
matmul = ufunc("matmul", 2, 1)
divide = ufunc("divide", 2, 1)
logaddexp = ufunc("logaddexp", 2, 1)
logaddexp2 = ufunc("logaddexp2", 2, 1)
true_divide = ufunc("true_divide", 2, 1)
floor_divide = ufunc("floor_divide", 2, 1)
negative = ufunc("negative", 1, 1)
positive = ufunc("positive", 1, 1)
power = ufunc("power", 2, 1)
remainder = ufunc("remainder", 2, 1)
mod = ufunc("mod", 2, 1)
divmod = ufunc("divmod", 2, 2)
absolute = ufunc("absolute", 1, 1)
fabs = ufunc("fabs", 1, 1)
rint = ufunc("rint", 1, 1)
sign = ufunc("sign", 1, 1)
heaviside = ufunc("heaviside", 1, 1)
conj = ufunc("conj", 1, 1)
exp = ufunc("exp", 1, 1)
exp2 = ufunc("exp2", 1, 1)
log = ufunc("log", 1, 1)
log2 = ufunc("log2", 1, 1)
log10 = ufunc("log10", 1, 1)
expm1 = ufunc("expm1", 1, 1)
log1p = ufunc("log1p", 1, 1)
sqrt = ufunc("sqrt", 1, 1)
square = ufunc("square", 1, 1)
cbrt = ufunc("cbrt", 1, 1)
reciprocal = ufunc("reciprocal", 1, 1)
gcd = ufunc("gcd", 1, 1)
lcm = ufunc("lcm", 1, 1)

# Trigonometric functions
sin = ufunc("sin", 1, 1)
cos = ufunc("cos", 1, 1)
tan = ufunc("tan", 1, 1)
arcsin = ufunc("arcsin", 1, 1)
arccos = ufunc("arccos", 1, 1)
arctan = ufunc("arctan", 1, 1)
arctan2 = ufunc("arctan2", 2, 1)
hypot = ufunc("hypot", 2, 1)
sinh = ufunc("sinh", 1, 1)
cosh = ufunc("cosh", 1, 1)
tanh = ufunc("tanh", 1, 1)
arcsinh = ufunc("arcsinh", 1, 1)
arccosh = ufunc("arccosh", 1, 1)
arctanh = ufunc("arctanh", 1, 1)
deg2rad = ufunc("deg2rad", 1, 1)
rad2deg = ufunc("rad2deg", 1, 1)

# Bit-twiddling functions
bitwise_and = ufunc("bitwise_and", 2, 1)
bitwise_or = ufunc("bitwise_or", 2, 1)
bitwise_xor = ufunc("bitwise_xor", 2, 1)
invert = ufunc("invert", 1, 1)
left_shift = ufunc("left_shift", 2, 1)
right_shift = ufunc("right_shift", 2, 1)

# Comparison functions
greater = ufunc("greater", 2, 1)
greater_equal = ufunc("greater_equal", 2, 1)
less = ufunc("less", 2, 1)
less_equal = ufunc("less_equal", 2, 1)
not_equal = ufunc("not_equal", 2, 1)
equal = ufunc("equal", 2, 1)
logical_and = ufunc("logical_and", 2, 1)
logical_or = ufunc("logical_or", 2, 1)
logical_xor = ufunc("logical_xor", 2, 1)
logical_not = ufunc("logical_not", 1, 1)
maximum = ufunc("maximum", 2, 1)
minimum = ufunc("minimum", 2, 1)
fmax = ufunc("fmax", 2, 1)
fmin = ufunc("fmin", 2, 1)

# Floating functions
isfinite = ufunc("isfinite", 1, 1)
isinf = ufunc("greater_equal", 1, 1)
isnan = ufunc("isnan", 1, 1)
isnat = ufunc("isnat", 1, 1)
signbit = ufunc("signbit", 1, 1)
copysign = ufunc("copysign", 2, 1)
nextafter = ufunc("nextafter", 2, 1)
spacing = ufunc("spacing", 1, 1)
modf = ufunc("modf", 1, 2)
ldexp = ufunc("ldexp", 2, 1)
frexp = ufunc("frexp", 1, 2)
fmod = ufunc("fmod", 2, 1)
floor = ufunc("floor", 1, 1)
ceil = ufunc("ceil", 1, 1)
trunc = ufunc("trunc", 1, 1)


@create_numpy(_dtype_argreplacer)
def full(shape, fill_value, dtype=None, order="C"):
    return (mark_dtype(dtype),)


@create_numpy(_dtype_argreplacer)
def arange(start, stop=None, step=None, dtype=None):
    return (mark_dtype(dtype),)


@create_numpy(_dtype_argreplacer)
def array(object, dtype=None, copy=True, order="K", subok=False, ndmin=0):
    return (mark_dtype(dtype),)


@create_numpy(
    _dtype_argreplacer,
    default=lambda shape, dtype, order="C": full(shape, 0, dtype, order),
)
def zeros(shape, dtype=float, order="C"):
    return (mark_dtype(dtype),)


@create_numpy(
    _dtype_argreplacer,
    default=lambda shape, dtype, order="C": full(shape, 1, dtype, order),
)
def ones(shape, dtype=float, order="C"):
    return (mark_dtype(dtype),)


@create_numpy(_dtype_argreplacer)
def eye(N, M=None, k=0, dtype=float, order="C"):
    return (mark_dtype(dtype),)


@create_numpy(_dtype_argreplacer)
def asarray(a, dtype=None, order=None):
    return (mark_dtype(dtype),)


def reduce_impl(red_ufunc: ufunc):
    def inner(a, **kwargs):
        return red_ufunc.reduce(a, **kwargs)

    return inner


@create_numpy(_reduce_argreplacer, default=reduce_impl(globals()["add"]))
@all_of_type(ndarray)
def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer, default=reduce_impl(globals()["multiply"]))
@all_of_type(ndarray)
def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer, default=reduce_impl(globals()["minimum"]))
@all_of_type(ndarray)
def min(a, axis=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer, default=reduce_impl(globals()["maximum"]))
@all_of_type(ndarray)
def max(a, axis=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer, default=reduce_impl(globals()["logical_or"]))
@all_of_type(ndarray)
def any(a, axis=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer, default=reduce_impl(globals()["logical_and"]))
@all_of_type(ndarray)
def all(a, axis=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer)
@all_of_type(ndarray)
def argmin(a, axis=None, out=None):
    return (a, mark_non_coercible(out))


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def nanargmin(a, axis=None):
    return (a,)


@create_numpy(_reduce_argreplacer)
@all_of_type(ndarray)
def argmax(a, axis=None, out=None):
    return (a, mark_non_coercible(out))


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def nanargmax(a, axis=None):
    return (a,)


@create_numpy(_reduce_argreplacer)
@all_of_type(ndarray)
def nanmin(a, axis=None, out=None):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer)
@all_of_type(ndarray)
def nanmax(a, axis=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer)
@all_of_type(ndarray)
def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer)
@all_of_type(ndarray)
def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer)
@all_of_type(ndarray)
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer)
@all_of_type(ndarray)
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    return (a, mark_non_coercible(out))


# set routines
@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=None):
    return (a,)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def in1d(element, test_elements, assume_unique=False, invert=False):
    return (element, test_elements)


def _isin_default(element, test_elements, assume_unique=False, invert=False):
    return in1d(
        element, test_elements, assume_unique=assume_unique, invert=invert
    ).reshape(element.shape)


@create_numpy(_first2argreplacer, default=_isin_default)
@all_of_type(ndarray)
def isin(element, test_elements, assume_unique=False, invert=False):
    return (element, test_elements)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    return (ar1, ar2)


def _setdiff1d_default(ar1, ar2, assume_unique=False):
    if assume_unique:
        ar1 = asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]


@create_numpy(_first2argreplacer, default=_setdiff1d_default)
@all_of_type(ndarray)
def setdiff1d(ar1, ar2, assume_unique=False):
    return (ar1, ar2)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def setxor1d(ar1, ar2, assume_unique=False):
    return (ar1, ar2)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def union1d(ar1, ar2):
    return (ar1, ar2)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def sort(a, axis=None, kind=None, order=None):
    return (a,)


def _tuple_check_argreplacer(args, kwargs, arrays):
    if len(arrays) == 1:
        return arrays + args[1:], kwargs
    else:
        return (arrays,) + args[1:], kwargs


@create_numpy(_tuple_check_argreplacer)
@all_of_type(ndarray)
def lexsort(keys, axis=None):
    if isinstance(keys, tuple):
        return keys
    else:
        return (keys,)


def _args_argreplacer(args, kwargs, arrays):
    return arrays, kwargs


@create_numpy(_args_argreplacer)
@all_of_type(ndarray)
def broadcast_arrays(*args, subok=False):
    return args


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def broadcast_to(array, shape, subok=False):
    return (array,)


def _first_argreplacer(args, kwargs, arrays1):
    def func(arrays, *args, **kwargs):
        return (arrays1,) + args, kwargs

    return func(*args, **kwargs)


@create_numpy(_first_argreplacer)
@all_of_type(ndarray)
def concatenate(arrays, axis=0, out=None):
    return arrays


@create_numpy(_first_argreplacer)
@all_of_type(ndarray)
def stack(arrays, axis=0, out=None):
    return arrays


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def argsort(a, axis=-1, kind="quicksort", order=None):
    return (a,)


@create_numpy(_self_argreplacer, default=lambda a: sort(a, axis=0))
@all_of_type(ndarray)
def msort(a):
    return (a,)


@create_numpy(_self_argreplacer, default=lambda a: sort(a))
@all_of_type(ndarray)
def sort_complex(a):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def partition(a, kth, axis=-1, kind="introselect", order=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def argpartition(a, kth, axis=-1, kind="introselect", order=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def nonzero(a):
    return (a,)


@create_numpy(_self_argreplacer, default=method_impl("transpose"))
@all_of_type(ndarray)
def transpose(a, axes=None):
    return (a,)


@create_numpy(_self_argreplacer, default=lambda a: transpose(nonzero(a)))
@all_of_type(ndarray)
def argwhere(a):
    return (a,)


@create_numpy(_self_argreplacer, default=method_impl("ravel"))
@all_of_type(ndarray)
def ravel(a):
    return (a,)


@create_numpy(_self_argreplacer, default=lambda a: nonzero(ravel(a))[0])
@all_of_type(ndarray)
def flatnonzero(a):
    return (a,)


def _where_def(condition, x=None, y=None):
    if x is None and y is None:
        return nonzero(condition)

    return NotImplemented


def _where_replacer(a, kw, d):
    def where_rd(condition, x=None, y=None):
        if d[1] is not None or d[2] is not None:
            return d, {}
        return (d[0],), {}

    return where_rd(*a, **kw)


@create_numpy(_where_replacer, default=_where_def)
@all_of_type(ndarray)
def where(condition, x=None, y=None):
    return (condition, x, y)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def pad(array, pad_width, mode, **kwargs):
    return (array,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def searchsorted(a, v, side="left", sorter=None):
    return (a,)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def compress(condition, a, axis=None, out=None):
    return (condition, a, out)


@create_numpy(
    _first2argreplacer,
    default=lambda condition, arr: compress(ravel(condition), ravel(arr)),
)
@all_of_type(ndarray)
def extract(condition, arr):
    return (condition, arr)


@create_numpy(
    _self_argreplacer, default=lambda a, axis=None: sum(a.astype("bool"), axis=axis)
)
@all_of_type(ndarray)
def count_nonzero(a, axis=None):
    return (a,)


class errstate:
    @create_numpy(_identity_argreplacer)
    def __new__(cls, **kwargs):
        return ()

    def __init__(cls, self):
        pass

    @create_numpy(_identity_argreplacer)
    def __enter__(self):
        return ()

    @create_numpy(_identity_argreplacer)
    def __exit__(self, exc_type, exc_value, exc_traceback):
        return ()


ufunc_list = []
for key, val in globals().copy().items():
    if isinstance(val, ufunc):
        ufunc_list.append(key)


@create_numpy(_self_argreplacer, default=getattr_impl("shape"))
@all_of_type(ndarray)
def shape(array):
    return (array,)


@create_numpy(_self_argreplacer, default=lambda array: len(shape(array)))
@all_of_type(ndarray)
def ndim(array):
    return (array,)


@create_numpy(
    _self_argreplacer,
    default=lambda array: functools.reduce(operator.mul, shape(array), 1),
)
@all_of_type(ndarray)
def size(array):
    return (array,)


@create_numpy(_self_argreplacer, default=getattr_impl("nbytes"))
@all_of_type(ndarray)
def nbytes(array):
    return (array,)


def _swapaxes_default(a, axis1, axis2):
    axes = list(range(ndim(a)))
    axes[axis1] = axis2
    axes[axis2] = axis1
    return transpose(a, tuple(axes))


@create_numpy(_self_argreplacer, default=_swapaxes_default)
@all_of_type(ndarray)
def swapaxes(a, axis1, axis2):
    return (a,)


def _moveaxis_default(a, source, destination):
    axes = list(range(ndim(a)))
    axes.remove(source)
    axes.insert(destination, source)
    return transpose(a, tuple(axes))


@create_numpy(_self_argreplacer, default=_moveaxis_default)
@all_of_type(ndarray)
def rollaxis(a, axis, start=0):
    return (a,)


@create_numpy(_self_argreplacer, default=_moveaxis_default)
@all_of_type(ndarray)
def moveaxis(a, source, destination):
    return (a,)


@create_numpy(_self_argreplacer, default=method_impl("reshape"))
@all_of_type(ndarray)
def reshape(a, newshape, order="C"):
    return (a,)


def _atleast_xd(*arys, min_dims=0):
    outs = []
    for a in arys:
        dims = ndim(a)
        missing_dims = min_dims - dims
        if missing_dims <= 0:
            outs.append(a)
            continue

        outs.append(a[(None,) * missing_dims])

    if len(outs) == 1:
        return outs[0]

    return tuple(outs)


@create_numpy(_first_argreplacer, default=functools.partial(_atleast_xd, min_dims=1))
@all_of_type(ndarray)
def atleast_1d(*arys):
    return arys


@create_numpy(_first_argreplacer, default=functools.partial(_atleast_xd, min_dims=2))
@all_of_type(ndarray)
def atleast_2d(*arys):
    return arys


@create_numpy(_first_argreplacer, default=functools.partial(_atleast_xd, min_dims=3))
@all_of_type(ndarray)
def atleast_3d(*arys):
    return arys


def _column_stack_default(tup):
    tup = list(tup)
    for i in range(len(tup)):
        dims = ndim(tup[i])
        if 1 <= dims <= 2:
            tup[i] = _atleast_xd(tup[i], min_dims=2)
        else:
            raise ValueError("Only 1D or 2D arrays expected.")

    return concatenate(tup, axis=1)


@create_numpy(_first_argreplacer, default=_column_stack_default)
@all_of_type(ndarray)
def column_stack(tup):
    return tup


def _hstack_default(tup):
    if builtins.all(ndim(a) == 1 for a in tup):
        return concatenate(tup)

    return concatenate(tup, axis=1)


@create_numpy(_first_argreplacer, default=_hstack_default)
@all_of_type(ndarray)
def hstack(tup):
    return tup


def _vstack_default(tup):
    tup = tuple(reshape(a, (1, shape(a)[0])) if ndim(a) == 1 else a for a in tup)
    return concatenate(tup)


@create_numpy(_first_argreplacer, default=_vstack_default)
@all_of_type(ndarray)
def vstack(tup):
    return tup


class _Recurser(object):
    def __init__(self, recurse_if):
        self.recurse_if = recurse_if

    def map_reduce(
        self,
        x,
        f_map=lambda x, **kwargs: x,
        f_reduce=lambda x, **kwargs: x,
        f_kwargs=lambda **kwargs: kwargs,
        **kwargs
    ):
        def f(x, **kwargs):
            if not self.recurse_if(x):
                return f_map(x, **kwargs)
            else:
                next_kwargs = f_kwargs(**kwargs)
                return f_reduce((f(xi, **next_kwargs) for xi in x), **kwargs)

        return f(x, **kwargs)

    def walk(self, x, index=()):
        do_recurse = self.recurse_if(x)
        yield index, x, do_recurse

        if not do_recurse:
            return
        for i, xi in enumerate(x):
            # yield from ...
            for v in self.walk(xi, index + (i,)):
                yield v


def _block_default(arrays):
    rec = _Recurser(recurse_if=lambda x: type(x) is list)

    list_ndim = None
    any_empty = False
    for index, value, entering in rec.walk(arrays):
        if type(value) is tuple:
            # not strictly necessary, but saves us from:
            #  - more than one way to do things - no point treating tuples like
            #    lists
            #  - horribly confusing behaviour that results when tuples are
            #    treated like ndarray
            raise TypeError(
                "{} is a tuple. "
                "Only lists can be used to arrange blocks, and np.block does "
                "not allow implicit conversion from tuple to ndarray.".format(index)
            )
        if not entering:
            curr_depth = len(index)
        elif len(value) == 0:
            curr_depth = len(index) + 1
            any_empty = True
        else:
            continue

        if list_ndim is not None and list_ndim != curr_depth:
            raise ValueError(
                "List depths are mismatched. First element was at depth {}, "
                "but there is an element at depth {} ({})".format(
                    list_ndim, curr_depth, index
                )
            )
        list_ndim = curr_depth

        # convert all the arrays to ndarrays
        arrays = rec.map_reduce(arrays, f_map=asarray, f_reduce=list)

        elem_ndim = rec.map_reduce(arrays, f_map=lambda xi: xi.ndim, f_reduce=max)
        ndim = builtins.max(list_ndim, elem_ndim)
        first_axis = ndim - list_ndim
        arrays = rec.map_reduce(
            arrays, f_map=lambda xi: _atleast_xd(xi, ndim), f_reduce=list
        )

        return rec.map_reduce(
            arrays,
            f_reduce=lambda xs, axis: concatenate(list(xs), axis=axis),
            f_kwargs=lambda axis: dict(axis=axis + 1),
            axis=first_axis,
        )


def _block_arg_extractor(arrays):
    if isinstance(arrays, list):
        for arr in arrays:
            yield from _block_arg_extractor(arr)
        return

    yield arrays


def _block_argreplacer(args, kwargs, d):
    d = iter(d)

    def block(arrays):
        if isinstance(arrays, list):
            return list(block(arr) for arr in arrays)

        return next(d)

    return (block(*args, **kwargs),), {}


@create_numpy(_block_argreplacer)
@all_of_type(ndarray)
def block(arrays):
    yield from _block_arg_extractor(arrays)
