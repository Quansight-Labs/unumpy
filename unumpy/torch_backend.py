import unumpy
import torch
from uarray import Dispatchable, wrap_single_convertor
import unumpy
from unumpy import ufunc, ufunc_list, ndarray

__ua_domain__ = "numpy"


def asarray(a, dtype=None, order=None):
    if torch.is_tensor(a):
        if dtype is not None and a.dtype != dtype:
            ret = a.clone()
            if a.requires_grad:
                ret = ret.requires_grad_()
            return ret

        return a
    try:
        import numpy as np

        if isinstance(a, np.ndarray):
            return torch.from_numpy(a)
    except ImportError:
        pass

    return torch.tensor(a, dtype=dtype)


_implementations = {
    unumpy.ufunc.__call__: lambda x, *a, **kw: x(*a, **kw),
    unumpy.asarray: asarray,
    unumpy.array: torch.Tensor,
    unumpy.arange: lambda start, stop, step, **kwargs: torch.arange(
        start, stop, step, **kwargs
    ),
}


def __ua_function__(method, args, kwargs):
    if method in _implementations:
        return _implementations[method](*args, **kwargs)

    if not hasattr(torch, method.__name__):
        return NotImplemented

    return getattr(torch, method.__name__)(*args, **kwargs)


@wrap_single_convertor
def __ua_convert__(value, dispatch_type, coerce):
    if dispatch_type is ufunc and value in _ufunc_mapping:
        return _ufunc_mapping[value]

    if value is None:
        return None

    if dispatch_type is ndarray:
        if not coerce and not torch.is_tensor(value):
            return NotImplemented

        return asarray(value) if value is not None else None

    return value


_ufunc_mapping = {}


for ufunc_name in ufunc_list:
    if ufunc_name.startswith("arc"):
        torch_name = ufunc_name.replace("arc", "a")
    else:
        torch_name = ufunc_name

    if hasattr(torch, torch_name):
        _ufunc_mapping[getattr(unumpy, ufunc_name)] = getattr(torch, torch_name)
