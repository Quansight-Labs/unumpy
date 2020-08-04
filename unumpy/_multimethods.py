import functools
import itertools
import collections
import numbers
import operator
from uarray import create_multimethod, mark_as, all_of_type, Dispatchable
import builtins

create_numpy = functools.partial(create_multimethod, domain="numpy")

e = 2.718281828459045235360287471352662498
pi = 3.141592653589793238462643383279502884
euler_gamma = 0.577215664901532860606512090082402431
nan = float("nan")
inf = float("inf")
NINF = float("-inf")
PZERO = 0.0
NZERO = -0.0
newaxis = None
NaN = NAN = nan
Inf = Infinity = PINF = infty = inf


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


class ClassOverrideMeta(type):
    def __new__(cls, name, bases, namespace):
        bases_new = []
        subclass = False
        for b in bases:
            if isinstance(b, cls):
                subclass = True
                bases_new.append(b._unwrapped)
            else:
                bases_new.append(b)

        if subclass:
            return type(name, tuple(bases_new), namespace)

        return super().__new__(cls, name, bases, namespace)

    def __init__(self, name, bases, namespace):
        self._unwrapped = type(name, bases, namespace)
        return super().__init__(name, bases, namespace)

    @property  # type: ignore
    @create_numpy(_identity_argreplacer, default=lambda self: self._unwrapped)
    def overridden_class(self):
        return ()

    @create_numpy(
        _identity_argreplacer,
        default=lambda self, value: isinstance(value, self.overridden_class),
    )
    def __instancecheck__(self, value):
        return ()

    @create_numpy(
        _identity_argreplacer,
        default=lambda self, value: issubclass(value, self.overridden_class),
    )
    def __subclasscheck__(self, value):
        return ()


class ClassOverrideMetaWithConstructor(ClassOverrideMeta):
    @create_numpy(
        _identity_argreplacer,
        default=lambda self, *a, **kw: self.overridden_class(*a, **kw),
    )
    def __call__(self, *args, **kwargs):
        return ()


class ClassOverrideMetaWithGetAttr(ClassOverrideMeta):
    @create_numpy(
        _identity_argreplacer,
        default=lambda self, name: getattr(self.overridden_class, name),
    )
    def __getattr__(self, name):
        return ()


class ClassOverrideMetaWithConstructorAndGetAttr(
    ClassOverrideMetaWithConstructor, ClassOverrideMetaWithGetAttr
):
    pass


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
    def func(a, b, *args, **kw):
        kw_out = kw.copy()
        if "out" in kw:
            kw_out["out"] = arrays[2]
        return arrays[:2] + args, kw_out

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

    return func


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


class ndarray(metaclass=ClassOverrideMetaWithConstructorAndGetAttr):
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


class dtype(metaclass=ClassOverrideMetaWithConstructorAndGetAttr):
    pass


class ufunc(metaclass=ClassOverrideMeta):
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
float_power = ufunc("float_power", 2, 1)
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
conjugate = ufunc("conjugate", 1, 1)
conj = conjugate
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
degrees = ufunc("degrees", 1, 1)
radians = ufunc("radians", 1, 1)
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
isinf = ufunc("isinf", 1, 1)
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
def empty(shape, dtype="float64", order="C"):
    return (mark_dtype(dtype),)


def _self_dtype_argreplacer(args, kwargs, dispatchables):
    def replacer(a, *args, dtype=None, **kwargs):
        out_kw = kwargs.copy()
        out_kw["dtype"] = dispatchables[1]

        return (dispatchables[0],) + args, out_kw

    return replacer(*args, **kwargs)


def _empty_like_default(prototype, dtype=None, order="K", subok=True, shape=None):
    if order != "K" or subok != True:
        return NotImplemented

    out_shape = _shape(prototype) if shape is None else shape
    out_dtype = prototype.dtype if dtype is None else dtype

    return empty(out_shape, dtype=out_dtype)


@create_numpy(_self_dtype_argreplacer, default=_empty_like_default)
@all_of_type(ndarray)
def empty_like(prototype, dtype=None, order="K", subok=True, shape=None):
    return (prototype, mark_dtype(dtype))


@create_numpy(_dtype_argreplacer)
def full(shape, fill_value, dtype=None, order="C"):
    return (mark_dtype(dtype),)


def _full_like_default(a, fill_value, dtype=None, order="K", subok=True, shape=None):
    if order != "K" or subok != True:
        return NotImplemented

    out_shape = _shape(a) if shape is None else shape
    out_dtype = a.dtype if dtype is None else dtype

    return full(out_shape, fill_value, dtype=out_dtype)


@create_numpy(_self_dtype_argreplacer, default=_full_like_default)
@all_of_type(ndarray)
def full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None):
    return (a, mark_dtype(dtype))


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
    _self_dtype_argreplacer,
    default=lambda a, dtype=None, order="K", subok=True, shape=None: full_like(
        a, 0, dtype, order, subok, shape
    ),
)
@all_of_type(ndarray)
def zeros_like(a, dtype=None, order="K", subok=True, shape=None):
    return (a, mark_dtype(dtype))


@create_numpy(
    _dtype_argreplacer,
    default=lambda shape, dtype, order="C": full(shape, 1, dtype, order),
)
def ones(shape, dtype=float, order="C"):
    return (mark_dtype(dtype),)


@create_numpy(
    _self_dtype_argreplacer,
    default=lambda a, dtype=None, order="K", subok=True, shape=None: full_like(
        a, 1, dtype, order, subok, shape
    ),
)
@all_of_type(ndarray)
def ones_like(a, dtype=None, order="K", subok=True, shape=None):
    return (a, mark_dtype(dtype))


@create_numpy(_dtype_argreplacer)
def eye(N, M=None, k=0, dtype=float, order="C"):
    return (mark_dtype(dtype),)


@create_numpy(_dtype_argreplacer, default=lambda n, dtype=None: eye(n, dtype=dtype))
def identity(n, dtype=None):
    return (mark_dtype(dtype),)


@create_numpy(_dtype_argreplacer)
def asarray(a, dtype=None, order=None):
    return (mark_dtype(dtype),)


@create_numpy(_self_dtype_argreplacer)
@all_of_type(ndarray)
def asanyarray(a, dtype=None, order=None):
    return (a, mark_dtype(dtype))


def _asfarray_default(a, dtype=float):
    a = asarray(a, dtype=dtype)
    if not a.dtype.name.startswith("float"):
        dtype = float

    return asarray(a, dtype=dtype)


@create_numpy(_dtype_argreplacer, default=_asfarray_default)
def asfarray(a, dtype=float):
    return (mark_dtype(dtype),)


@create_numpy(
    _dtype_argreplacer,
    default=lambda a, dtype=None: asarray(a, dtype=dtype, order="F"),
)
def asfortranarray(a, dtype=None):
    return (mark_dtype(dtype),)


def _asarray_chkfinite_default(a, dtype=None, order=None):
    arr = asarray(a, dtype=dtype, order=order)
    if not all(isfinite(arr)):
        raise ValueError("Array must not contain infs or NaNs.")

    return arr


@create_numpy(_dtype_argreplacer, default=_asarray_chkfinite_default)
def asarray_chkfinite(a, dtype=None, order=None):
    return (mark_dtype(dtype),)


@create_numpy(_self_dtype_argreplacer)
@all_of_type(ndarray)
def require(a, dtype=None, requirements=None):
    return (a, mark_dtype(dtype))


@create_numpy(
    _dtype_argreplacer,
    default=lambda a, dtype=None: asarray(a, dtype=dtype, order="C"),
)
@all_of_type(ndarray)
def ascontiguousarray(a, dtype=None):
    return (mark_dtype(dtype),)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def copy(a, order="K"):
    return (a,)


@create_numpy(_dtype_argreplacer)
def frombuffer(buffer, dtype=float, count=-1, offset=0):
    return (mark_dtype(dtype),)


@create_numpy(_dtype_argreplacer)
def fromfile(file, dtype=float, count=-1, sep="", offset=0):
    return (mark_dtype(dtype),)


@create_numpy(_dtype_argreplacer)
def fromfunction(function, shape, **kwargs):
    if "dtype" in kwargs:
        dtype = kwargs["dtype"]
    else:
        dtype = float

    return (mark_dtype(dtype),)


def _fromiter_default(iterable, dtype, count=-1):
    if not isinstance(iterable, collections.abc.Iterable):
        raise TypeError("'%s' object is not iterable" % type(iterable).__name__)
    if count >= 0:
        iterable = itertools.islice(iterable, 0, count)

    return array(list(iterable), dtype=dtype)


@create_numpy(_dtype_argreplacer, default=_fromiter_default)
def fromiter(iterable, dtype, count=-1):
    return (mark_dtype(dtype),)


@create_numpy(_dtype_argreplacer)
def fromstring(string, dtype=float, count=-1, sep=""):
    return (mark_dtype(dtype),)


@create_numpy(_dtype_argreplacer)
def loadtxt(
    fname,
    dtype=float,
    comments="#",
    delimiter=None,
    converters=None,
    skiprows=0,
    usecols=None,
    unpack=False,
    ndmin=0,
    encoding="bytes",
    max_rows=None,
):
    return (mark_dtype(dtype),)


def reduce_impl(red_ufunc: ufunc):
    def inner(a, **kwargs):
        return red_ufunc.reduce(a, **kwargs)

    return inner


@create_numpy(_reduce_argreplacer, default=reduce_impl(add))
@all_of_type(ndarray)
def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer, default=reduce_impl(multiply))
@all_of_type(ndarray)
def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer, default=reduce_impl(minimum))
@all_of_type(ndarray)
def min(a, axis=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer, default=reduce_impl(maximum))
@all_of_type(ndarray)
def max(a, axis=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer, default=reduce_impl(logical_or))
@all_of_type(ndarray)
def any(a, axis=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer, default=reduce_impl(logical_and))
@all_of_type(ndarray)
def all(a, axis=None, out=None, keepdims=False):
    return (a, mark_non_coercible(out))


def _self_out_argreplacer(args, kwargs, dispatchables):
    def replacer(a, *args, out=None, **kwargs):
        return (dispatchables[0],) + args, {"out": dispatchables[1], **kwargs}

    return replacer(*args, **kwargs)


@create_numpy(_self_out_argreplacer, default=lambda x, out=None: equal(x, inf, out=out))
@all_of_type(ndarray)
def isposinf(x, out=None):
    return (x, mark_non_coercible(out))


@create_numpy(
    _self_out_argreplacer, default=lambda x, out=None: equal(x, NINF, out=out)
)
@all_of_type(ndarray)
def isneginf(x, out=None):
    return (x, mark_non_coercible(out))


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def iscomplex(x):
    return (x,)


@create_numpy(_identity_argreplacer)
def iscomplexobj(x):
    return ()


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def isreal(x):
    return (x,)


@create_numpy(_identity_argreplacer)
def isrealobj(x):
    return ()


@create_numpy(_identity_argreplacer)
def isscalar(element):
    return ()


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


def _self_dtype_out_argreplacer(args, kwargs, dispatchables):
    def replacer(a, *args, dtype=None, out=None, **kwargs):
        return (
            (dispatchables[0],) + args,
            dict(dtype=dispatchables[1], out=dispatchables[2], **kwargs),
        )

    return replacer(*args, **kwargs)


@create_numpy(_self_dtype_out_argreplacer)
@all_of_type(ndarray)
def cumprod(a, axis=None, dtype=None, out=None):
    return (a, mark_dtype(dtype), mark_non_coercible(out))


@create_numpy(_self_dtype_out_argreplacer)
@all_of_type(ndarray)
def cumsum(a, axis=None, dtype=None, out=None):
    return (a, mark_dtype(dtype), mark_non_coercible(out))


@create_numpy(
    _self_dtype_out_argreplacer,
    default=lambda a, axis=None, dtype=None, out=None: cumprod(
        where(isnan(a), 1, a), axis=axis, dtype=dtype, out=out  # type: ignore
    ),
)
@all_of_type(ndarray)
def nancumprod(a, axis=None, dtype=None, out=None):
    return (a, mark_dtype(dtype), mark_non_coercible(out))


@create_numpy(
    _self_dtype_out_argreplacer,
    default=lambda a, axis=None, dtype=None, out=None: cumsum(
        where(isnan(a), 0, a), axis=axis, dtype=dtype, out=out  # type: ignore
    ),
)
@all_of_type(ndarray)
def nancumsum(a, axis=None, dtype=None, out=None):
    return (a, mark_dtype(dtype), mark_non_coercible(out))


@create_numpy(_reduce_argreplacer)
@all_of_type(ndarray)
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    return (a, mark_non_coercible(out))


@create_numpy(_reduce_argreplacer)
@all_of_type(ndarray)
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    return (a, mark_non_coercible(out))


def _ptp_default(a, axis=None, out=None, keepdims=False):
    result = max(a, axis=axis, out=out, keepdims=keepdims)
    result -= min(a, axis=axis, out=None, keepdims=keepdims)
    return result


@create_numpy(_reduce_argreplacer, default=_ptp_default)
@all_of_type(ndarray)
def ptp(a, axis=None, out=None, keepdims=False):
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
    return reshape(
        in1d(element, test_elements, assume_unique=assume_unique, invert=invert),
        shape(element),
    )


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


def _normalize_axis(ndim, axis, allow_repeats=False):
    if not isinstance(axis, collections.abc.Sequence):
        axis = operator.index(axis)
        axis_pos = ndim + axis if axis < 0 else axis

        if axis_pos >= ndim or axis_pos < 0:
            raise ValueError(
                "Axis %s is out of bounds for array of dimension %s." % (axis, ndim)
            )

        return axis_pos

    axes = tuple(_normalize_axis(ndim, ax) for ax in axis)

    if not allow_repeats and (len(axes) != len(set(axes))):
        raise ValueError("Repeated axis.")

    return axes


def _expand_dims_default(a, axis):
    if isinstance(axis, numbers.Integral):
        axis = (axis,)

    axis = _normalize_axis(ndim(a) + len(axis), axis)

    out_shape = list(a.shape)

    for i, ax in enumerate(sorted(axis)):
        out_shape.insert(ax + i, 1)

    return a.reshape(tuple(out_shape))


@create_numpy(_self_argreplacer, default=_expand_dims_default)
@all_of_type(ndarray)
def expand_dims(a, axis):
    return (a,)


def _squeeze_default(a, axis=None):
    if axis is None:
        out_shape = tuple(dim for dim in a.shape if dim != 1)

        return a.reshape(out_shape)

    if isinstance(axis, numbers.Number):
        axis = (axis,)

    axis = _normalize_axis(ndim(a), axis)

    out_shape = list(a.shape)

    for i, ax in enumerate(sorted(axis)):
        if a.shape[ax] == 1:
            del out_shape[ax - i]
        else:
            raise ValueError(
                "Cannot select an axis to squeeze out which has size not equal to one."
            )

    return a.reshape(tuple(out_shape))


@create_numpy(_self_argreplacer, default=_squeeze_default)
@all_of_type(ndarray)
def squeeze(a, axis=None):
    return (a,)


def _meshgrid_default(*args, indexing="xy", sparse=False, copy=True):
    ndim = len(args)

    if indexing not in ["xy", "ij"]:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [
        reshape(asarray(x), s0[:i] + (-1,) + s0[i + 1 :]) for i, x in enumerate(args)
    ]

    if indexing == "xy" and ndim > 1:
        # switch first and second axis
        output = (
            reshape(output[0], (1, -1) + s0[2:]),
            reshape(output[1], (1, -1) + s0[2:]),
        )

    if not sparse:
        # Return the full N-D matrix (not only the 1-D vector)
        output = broadcast_arrays(*output, subok=True)

    if copy:
        output = [x.copy() for x in output]

    return output


@create_numpy(_args_argreplacer, default=_meshgrid_default)
@all_of_type(ndarray)
def meshgrid(*args, indexing="xy", sparse=False, copy=True):
    return args


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


@create_numpy(_self_argreplacer, default=method_impl("reshape"))
@all_of_type(ndarray)
def reshape(a, newshape, order="C"):
    return (a,)


@create_numpy(_self_argreplacer, default=lambda a: reshape(a, -1))
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


def _first2_dtype_argreplacer(args, kwargs, dispatchables):
    def replacer(a, b, *args, dtype=None, **kwargs):
        kw_out = kwargs.copy()
        kw_out["dtype"] = dispatchables[2]

        return dispatchables[:2] + args, kw_out

    return replacer(*args, **kwargs)


@create_numpy(_first2_dtype_argreplacer)
@all_of_type(ndarray)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    return (start, stop, mark_dtype(dtype))


def _logspace_default(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    return base ** linspace(
        start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis
    )


@create_numpy(_first2_dtype_argreplacer, default=_logspace_default)
@all_of_type(ndarray)
def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    return (start, stop, mark_dtype(dtype))


@create_numpy(_first2_dtype_argreplacer)
@all_of_type(ndarray)
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    return (start, stop, mark_dtype(dtype))


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
    if type(val) is ufunc:
        ufunc_list.append(key)


def _copyto_argreplacer(args, kwargs, dispatchables):
    def func(dst, src, casting="same_kind", where=True):
        return dispatchables[:2], dict(casting=casting, where=dispatchables[2])

    return func(*args, **kwargs)


@create_numpy(_copyto_argreplacer)
@all_of_type(ndarray)
def copyto(dst, src, casting="same_kind", where=True):
    return (mark_non_coercible(dst), src, where)


@create_numpy(_self_argreplacer, default=getattr_impl("shape"))
@all_of_type(ndarray)
def shape(array):
    return (array,)


_shape = shape


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


def _dstack_default(tup):
    arrays = []
    for arr in tup:
        nd = ndim(arr)
        if nd == 1:
            arrays.append(expand_dims(arr, (0, -1)))
        elif nd == 2:
            arrays.append(expand_dims(arr, -1))
        else:
            arrays.append(arr)

    return concatenate(arrays, axis=2)


@create_numpy(_first_argreplacer, default=_dstack_default)
@all_of_type(ndarray)
def dstack(tup):
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


def _diff_default(a, n=1, axis=-1):
    if n == 0:
        return a
    if n < 0:
        raise ValueError("order must be non-negative but got " + repr(n))

    a = asarray(a)
    nd = ndim(a)
    if nd == 0:
        raise ValueError("diff requires input that is at least one dimensional")
    if axis < -nd or axis >= nd:
        raise ValueError("axis out of range")
    axis = axis % nd

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    op = not_equal if a.dtype.kind == "b" else subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])

    return a


@create_numpy(_self_argreplacer, default=_diff_default)
@all_of_type(ndarray)
def diff(a, n=1, axis=-1):
    return (a,)


def _ediff1d_argreplacer(args, kwargs, dispatchables):
    return (
        (dispatchables[0],),
        dict(to_end=dispatchables[1], to_begin=dispatchables[2]),
    )


def _ediff1d_default(ary, to_end=None, to_begin=None):
    ary = ravel(ary)

    diffs = ary[1:] - ary[:-1]

    if to_end is None and to_begin is None:
        return diffs

    arrays = []
    if to_begin is not None:
        arrays.append(ravel(to_begin))

    arrays.append(diff)

    if to_end is not None:
        arrays.append(ravel(to_end))

    return concatenate(arrays)


@create_numpy(_ediff1d_argreplacer, default=_ediff1d_default)
@all_of_type(ndarray)
def ediff1d(ary, to_end=None, to_begin=None):
    return (ary, to_end, to_begin)


@create_numpy(_args_argreplacer)
@all_of_type(ndarray)
def gradient(a, *varargs, edge_order=1, axis=None):
    return (a,) + varargs


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    return (a, b)


def _trapz_argreplacer(args, kwargs, dispatchables):
    def replacer(y, x=None, dx=1.0, axis=-1):
        return (dispatchables[0],), dict(x=dispatchables[1], dx=dx, axis=axis)

    return replacer(*args, **kwargs)


@create_numpy(_trapz_argreplacer)
@all_of_type(ndarray)
def trapz(y, x=None, dx=1.0, axis=-1):
    return (y, x)


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
    import unumpy as np

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

        elem_ndim = rec.map_reduce(
            arrays, f_map=lambda xi: np.ndim(xi), f_reduce=builtins.max
        )
        ndim = builtins.max(list_ndim, elem_ndim)
        first_axis = ndim - list_ndim
        arrays = rec.map_reduce(
            arrays, f_map=lambda xi: _atleast_xd(xi, ndim), f_reduce=list
        )

        return rec.map_reduce(
            arrays,
            f_reduce=lambda xs, axis: concatenate(list(xs), axis=axis - 1),
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


@create_numpy(_block_argreplacer, default=_block_default)
@all_of_type(ndarray)
def block(arrays):
    yield from _block_arg_extractor(arrays)


def _isclose_default(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    ret = absolute(a - b) <= (atol + rtol * absolute(b))
    if equal_nan:
        ret |= isnan(a) & isnan(b)

    return ret


@create_numpy(_first2argreplacer, default=_isclose_default)
@all_of_type(ndarray)
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return a, b


def _allclose_default(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


@create_numpy(_first2argreplacer, default=_allclose_default)
@all_of_type(ndarray)
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return a, b


@create_numpy(
    _first2argreplacer, default=lambda a1, a2: shape(a1) == shape(a2) and all(a1 == a2)
)
@all_of_type(ndarray)
def array_equal(a1, a2):
    return a1, a2


@create_numpy(_first2argreplacer, default=lambda a1, a2: all(a1 == a2))
@all_of_type(ndarray)
def array_equiv(a1, a2):
    return a1, a2


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def diag(v, k=0):
    return (v,)


@create_numpy(_self_argreplacer, default=lambda v, k=0: diag(ravel(v), k))
@all_of_type(ndarray)
def diagflat(v, k=0):
    return (v,)


def _tri_default(N, M=None, k=0, dtype=float):
    if M is None:
        M = N

    ii, jj = arange(N).reshape(-1, 1), arange(M)
    mask = jj <= ii + k

    return array(where(mask, 1, 0), dtype=dtype)


@create_numpy(_dtype_argreplacer, default=_tri_default)
def tri(N, M=None, k=0, dtype=float):
    return (mark_dtype(dtype),)


def _tril_default(m, k=0):
    ii, jj = arange(m.shape[0]).reshape(-1, 1), arange(m.shape[1])
    mask = jj <= ii + k

    return array(where(mask, m, 0), dtype=m.dtype)


@create_numpy(_self_argreplacer, default=_tril_default)
@all_of_type(ndarray)
def tril(m, k=0):
    return (m,)


def _triu_default(m, k=0):
    ii, jj = arange(m.shape[0]).reshape(-1, 1), arange(m.shape[1])
    mask = jj >= ii + k

    return array(where(mask, m, 0), dtype=m.dtype)


@create_numpy(_self_argreplacer, default=_triu_default)
@all_of_type(ndarray)
def triu(m, k=0):
    return (m,)


def _vander_default(x, N=None, increasing=False):
    x = array(x)
    if N is None:
        N = len(x)

    arr = zeros((len(x), N)) + x.reshape((-1, 1))

    exps = arange(N)
    if increasing == False:
        exps = exps[::-1]

    return array(power(arr, exps), dtype=x.dtype)


@create_numpy(_self_argreplacer, default=_vander_default)
@all_of_type(ndarray)
def vander(x, N=None, increasing=False):
    return (x,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def split(ary, indices_or_sections, axis=0):
    return (ary,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def array_split(ary, indices_or_sections, axis=0):
    return (ary,)


@create_numpy(
    _self_argreplacer,
    default=lambda ary, indices_or_sections: split(ary, indices_or_sections, axis=2),
)
@all_of_type(ndarray)
def dsplit(ary, indices_or_sections):
    return (ary,)


@create_numpy(
    _self_argreplacer,
    default=lambda ary, indices_or_sections: split(ary, indices_or_sections, axis=1),
)
@all_of_type(ndarray)
def hsplit(ary, indices_or_sections):
    return (ary,)


@create_numpy(
    _self_argreplacer,
    default=lambda ary, indices_or_sections: split(ary, indices_or_sections, axis=0),
)
@all_of_type(ndarray)
def vsplit(ary, indices_or_sections):
    return (ary,)


def _tile_default(A, reps):
    if isinstance(reps, numbers.Integral):
        reps = (reps,)

    n = len(reps) - ndim(A)
    if n > 0:
        A = expand_dims(A, tuple(range(n)))
    elif n < 0:
        reps = ((1,) * -n) + tuple(reps)

    new_shape = tuple(dim * rep for dim, rep in zip(A.shape, reps))

    for axis, dim in enumerate(reps):
        dim = operator.index(dim)
        arrays = [A] * dim
        if len(arrays) == 0:
            return asarray(arrays, dtype=A.dtype).reshape(new_shape)

        A = concatenate(arrays, axis=axis)

    return A


@create_numpy(_self_argreplacer, default=_tile_default)
@all_of_type(ndarray)
def tile(A, reps):
    return (A,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def repeat(a, repeats, axis=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def delete(arr, obj, axis=None):
    return (arr,)


def _insert_argreplacer(args, kwargs, dispatchables):
    def replacer(arr, obj, values, axis=None):
        return (dispatchables[0], obj, dispatchables[1]), {"axis": axis}

    return replacer(*args, **kwargs)


@create_numpy(_insert_argreplacer)
@all_of_type(ndarray)
def insert(arr, obj, values, axis=None):
    return (arr, values)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def append(arr, values, axis=None):
    return (arr, values)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def resize(a, new_shape):
    return (a,)


def _trim_zeros_default(filt, trim="fb"):
    nonzero_idxs = nonzero(filt)[0]

    if len(nonzero_idxs) == 0:
        return asarray([], dtype=filt.dtype)

    start, end = None, None
    if "f" in trim:
        start = nonzero_idxs[0]
    if "b" in trim:
        end = nonzero_idxs[-1] + 1
    return filt[start:end]


@create_numpy(_self_argreplacer, default=_trim_zeros_default)
@all_of_type(ndarray)
def trim_zeros(filt, trim="fb"):
    return (filt,)


def _flip_default(m, axis=None):
    nd = ndim(m)
    if axis is None:
        axis = tuple(range(nd))
    else:
        axis = _normalize_axis(nd, axis)

        if not isinstance(axis, collections.abc.Sequence):
            axis = (axis,)

    slices = [slice(None, None, 1)] * nd
    for ax in axis:
        slices[ax] = slice(None, None, -1)

    return m[tuple(slices)]


@create_numpy(_self_argreplacer, default=_flip_default)
@all_of_type(ndarray)
def flip(m, axis=None):
    return (m,)


@create_numpy(_self_argreplacer, default=lambda m: flip(m, 1))
@all_of_type(ndarray)
def fliplr(m):
    return (m,)


@create_numpy(_self_argreplacer, default=lambda m: flip(m, 0))
@all_of_type(ndarray)
def flipud(m):
    return (m,)


def _roll_default(a, shift, axis=None):
    if axis is None:
        return _roll_default(ravel(a), shift, axis=0).reshape(a.shape)

    if isinstance(axis, numbers.Integral):
        axis = (axis,)
    if isinstance(shift, numbers.Integral):
        shift = (shift,)

    axis = _normalize_axis(ndim(a), axis, allow_repeats=True)

    if not len(axis) == len(shift):
        if len(shift) == 1:
            shift = itertools.repeat(shift[0], len(axis))
        elif len(axis) == 1:
            axis = itertools.repeat(axis[0], len(shift))
        else:
            raise ValueError("shift and axis must have the same length.")

    for s, ax in zip(shift, axis):
        dim = a.shape[ax]
        if s >= 0:
            index = dim - (s % dim)
        else:
            index = abs(s) % dim

        arrays = array_split(a, [index], axis=ax)

        a = concatenate(arrays[::-1], axis=ax)

    return a


@create_numpy(_self_argreplacer, default=_roll_default)
@all_of_type(ndarray)
def roll(a, shift, axis=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def rot90(m, k=1, axes=(0, 1)):
    return (m,)


def _apply_along_axis_argreplacer(args, kwargs, dispatchables):
    def replacer(func1d, axis, arr, *args, **kwargs):
        return (func1d, axis, dispatchables[0]) + args, kwargs

    return replacer(*args, **kwargs)


@create_numpy(_apply_along_axis_argreplacer)
@all_of_type(ndarray)
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    return (arr,)


def _apply_over_axes_argreplacer(args, kwargs, dispatchables):
    def replacer(func, a, axes):
        return (func, dispatchables[0], axes), dict()

    return replacer(*args, **kwargs)


def _apply_over_axes_default(func, a, axes):
    nd = ndim(a)
    axes = _normalize_axis(nd, axes)

    res = a
    for axis in axes:
        res = func(res, axis)

        res_nd = ndim(res)
        if res_nd < nd - 1:
            raise ValueError(
                "Function is not returning an array with the correct number of dimensions."
            )
        if res_nd == nd - 1:
            res = expand_dims(res, axis)

    return res


@create_numpy(_apply_over_axes_argreplacer, default=_apply_over_axes_default)
@all_of_type(ndarray)
def apply_over_axes(func, a, axes):
    return (a,)


@create_numpy(_identity_argreplacer)
def frompyfunc(func, nin, nout, identity=None):
    return ()


def _piecewise_default(x, condlist, funclist, *args, **kw):
    if not isinstance(condlist, list):
        condlist = [condlist]

    condlist = [asarray(cond) for cond in condlist]

    n1 = len(condlist)
    n2 = len(funclist)

    if n1 != n2:
        if n1 + 1 == n2:
            condelse = ~any(condlist, axis=0, keepdims=True)
            condlist = concatenate([condlist, condelse], axis=0)
        else:
            raise ValueError(
                "With %d condition(s), either %d or %d functions are expected."
                % (n, n, n + 1)
            )

    y = zeros(x.shape, dtype=x.dtype)

    for i, (cond, func) in enumerate(zip(condlist, funclist)):
        if cond.shape != x.shape and ndim(cond) != 0:
            raise ValueError(
                "Condition at index %d doesn't have the same shape as x." % i
            )

        if isinstance(func, collections.abc.Callable):
            y = where(cond, func(x, *args, **kw), y)
        else:
            y = where(cond, func, y)

    return y


@create_numpy(_self_argreplacer, default=_piecewise_default)
@all_of_type(ndarray)
def piecewise(x, condlist, funclist, *args, **kw):
    return (x,)


def _unwrap_default(p, discont=3.141592653589793, axis=-1):
    nd = ndim(p)

    dd = diff(p, axis=axis)

    slice0 = [slice(None)] * nd
    slice0[axis] = slice(0, 1)
    slice0 = tuple(slice0)

    slice1 = [slice(None, None)] * nd
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)

    ddmod = mod(dd + pi, 2 * pi) - pi
    ddmod = where((ddmod == -pi) & (dd > 0), pi, ddmod)

    ph_correct = ddmod - dd
    ph_correct = where(absolute(dd) < discont, 0, ph_correct)

    up_slice0 = p[slice0]
    up_slice1 = p[slice1] + cumsum(ph_correct, axis=axis)

    up = concatenate([up_slice0, up_slice1], axis=axis)

    return up


@create_numpy(_self_argreplacer, default=_unwrap_default)
@all_of_type(ndarray)
def unwrap(p, discont=3.141592653589793, axis=-1):
    return (p,)


@create_numpy(_self_out_argreplacer)
@all_of_type(ndarray)
def around(a, decimals=0, out=None):
    return (a, mark_non_coercible(out))


round_ = around


@create_numpy(_self_out_argreplacer, default=lambda x, out=None: trunc(x, out=out))
@all_of_type(ndarray)
def fix(x, out=None):
    return (x, mark_non_coercible(out))


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def i0(x):
    return (x,)


@create_numpy(
    _self_argreplacer, default=lambda x: where(x != 0, sin(pi * x) / (pi * x), 1)
)
@all_of_type(ndarray)
def sinc(x):
    return (x,)


def _angle_default(z, deg=False):
    angles = arctan2(z.imag, z.real)
    if deg:
        angles *= 180 / pi

    return angles


@create_numpy(_self_argreplacer, default=_angle_default)
@all_of_type(ndarray)
def angle(z, deg=False):
    return (z,)


@create_numpy(_self_argreplacer, default=lambda val: val.real)
@all_of_type(ndarray)
def real(val):
    return (val,)


@create_numpy(_self_argreplacer, default=lambda val: val.imag)
@all_of_type(ndarray)
def imag(val):
    return (val,)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def convolve(a, v, mode="full"):
    return (a, v)


def _clip_argreplacer(args, kwargs, dispatchables):
    def replacer(a, a_min, a_max, out=None, **kwargs):
        return dispatchables[:3], dict(out=dispatchables[3], **kwargs)

    return replacer(*args, **kwargs)


def _clip_default(a, a_min, a_max, out=None, **kwargs):
    if a_min is None and a_max is None:
        raise ValueError("One of max or min must be given.")

    if a_min is not None:
        a = where(a < a_min, a_min, a)
    if a_max is not None:
        a = where(a > a_max, a_max, a)

    if out is None:
        return a
    else:
        copyto(out, a)


@create_numpy(_clip_argreplacer, default=_clip_default)
@all_of_type(ndarray)
def clip(a, a_min, a_max, out=None, **kwargs):
    return (a, a_min, a_max, mark_non_coercible(out))


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    return (x,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def real_if_close(a, tol=100):
    return (a,)


def _interp_default(x, xp, fp, left=None, right=None, period=None):
    if ndim(xp) != 1 or ndim(fp) != 1:
        raise ValueError("Data points must be 1-D sequences.")
    if len(xp) != len(fp):
        raise ValueError("fp and xp are not of the same length.")

    sorted_idxs = argsort(xp)
    xp = xp[sorted_idxs]
    fp = fp[sorted_idxs]

    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value.")

        period = abs(period)
        x = x % period
        xp = xp % period

        xp = concatenate([xp[-1:] - period, xp, xp[0:1] + period])
        fp = concatenate([fp[-1:], fp, fp[0:1]])

    idxs = searchsorted(xp, x)
    idxs = where(idxs >= len(xp), len(xp) - 1, idxs)

    y0 = fp[idxs - 1]
    y1 = fp[idxs]
    x0_dist = x - xp[idxs - 1]
    x1_dist = xp[idxs] - x
    x_dist = xp[idxs] - xp[idxs - 1]

    result = (y0 * x1_dist + y1 * x0_dist) / x_dist

    left = fp[0] if left is None else left
    right = fp[-1] if right is None else right

    result = where(x < xp[0], left, result)
    result = where(x > xp[-1], right, result)

    return result


@create_numpy(_self_argreplacer, default=_interp_default)
@all_of_type(ndarray)
def interp(x, xp, fp, left=None, right=None, period=None):
    return (x,)
