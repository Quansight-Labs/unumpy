"""
.. note::
    If you are interested in writing backends or multimethods for ``unumpy``,
    please look at the documentation for :obj:`uarray`, which explains how to
    do this.

``unumpy`` is meant for three groups of individuals:

* Those who write array-like objects, like developers of Dask, Xnd, PyData/Sparse,
  CuPy, and others.
* Library authors or programmers that hope to target multiple array backends, listed
  above.
* Users who wish to target their code to other backends.

For example, the following is currently possible:

>>> import uarray as ua
>>> import unumpy as np
>>> from unumpy.dask_backend import DaskBackend
>>> import unumpy.sparse_backend as SparseBackend
>>> import sparse, dask.array as da
>>> def main():
...     x = np.zeros(5)
...     return np.exp(x)
>>> with ua.set_backend(DaskBackend()):
...     isinstance(main(), da.core.Array)
True
>>> with ua.set_backend(SparseBackend):
...     isinstance(main(), sparse.SparseArray)
True

Now imagine some arbitrarily nested code, all for which the implementations can be
switched out using a simple context manager.

``unumpy`` is an in-progress mirror of the NumPy API which allows the user
to dynamically switch out the backend that is used. It also allows
auto-selection of the backend based on the arguments passed into a function. It does this by
defining a collection of :obj:`uarray` multimethods that support dispatch.
Although it currently provides a number of backends, the aspiration is that,
with time, these back-ends will move into the respective libraries and it will be possible
to use the library modules directly as backends.

Note that currently, our coverage is very incomplete. However, we have attempted
to provide at least one of each kind of object in ``unumpy`` for
reference. There are :obj:`ufunc` s and :obj:`ndarray` s,  which are classes,
methods on :obj:`ufunc` such as :obj:`__call__ <ufunc.__call__>`, and
:obj:`reduce <ufunc.reduce>` and also functions such as :obj:`sum`.

Where possible, we attempt to provide default implementations so that the whole API
does not have to be reimplemented, however, it might be useful to gain speed or to
re-implement it in terms of other functions which already exist in your library.

The idea is that once things are more mature, it will be possible to switch
out your backend with a simple import statement switch:

.. code:: python

    import numpy as np  # Old method
    import unumpy as np  # Once this project is mature

Currently, the following functions are supported:

* All NumPy `universal functions <https://www.numpy.org/devdocs/reference/ufuncs.html>`_.

  * :obj:`ufunc reductions <numpy.ufunc.reduce>`

For the full range of functions, use ``dir(unumpy)``.

You can use the :obj:`uarray.set_backend` decorator to set a backend and use the
desired backend. Note that not every backend supports every method. For example,
PyTorch does not have an exact :obj:`ufunc` equivalent, so we dispatch to actual
methods using a dictionary lookup. The following
backends are supported:

* :obj:`numpy_backend`
* :obj:`torch_backend`
* :obj:`xnd_backend`
* :obj:`dask_backend`
* :obj:`cupy_backend`
* :obj:`sparse_backend`

Writing Backends
----------------

Since :obj:`unumpy` is based on :obj:`uarray`, all overrides are done via the ``__ua_*__``
protocols. We strongly recommend you read the
`uarray documentation <https://uarray.readthedocs.io/en/latest>`_ for context.

All functions/methods in :obj:`unumpy` are :obj:`uarray` multimethods. This means
you can override them using the ``__ua_function__`` protocol.

In addition, :obj:`unumpy` allows dispatch on :obj:`numpy.ndarray`,
:obj:`numpy.ufunc` and :obj:`numpy.dtype` via the ``__ua_convert__`` protocol.

Dispatching on objects means one can intercept these, convert to an equivalent
native format, or dispatch on their methods, including ``__call__``.

We suggest you browse the source for example backends.

Differences between overriding :obj:`numpy.ufunc` objects and other multimethods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Of note here is that there are certain callable objects within NumPy, most
prominently :obj:`numpy.ufunc` objects, that are not typical functions/methods,
and so cannot be directly overridden, the key word here being *directly*.

In Python, when a method is called, i.e. ``x.method(*a, **kw)`` it is the same
as writing ``type(x).method(x, *a, **kw)`` assuming that ``method`` was a regular
method defined on the type. This allows some very interesting things to happen.

For instance, if we make ``method`` a multimethod, it allows us to override
methods, provided we know that the first argument passed in will be ``x``.

One other thing that is possible (and done in :obj:`unumpy`) is to override the
``__call__`` method on a callable object. This is, in fact, exactly how to override
a :obj:`ufunc`.

Other interesting things that can be done (but as of now, are not) are to replace
:obj:`ufunc` objects entirely by native equivalents overriding the ``__get__`` method.
This technique can also be applied to ``dtype`` objects.

Meta-array support
^^^^^^^^^^^^^^^^^^

Meta-arrays are arrays that can hold other arrays, such as Dask arrays and XArray
datasets.

If meta-arrays and libraries depend on :obj:`unumpy` instead of NumPy, they can benefit
from containerization and hold arbitrary arrays; not just :obj:`numpy.ndarray` objects.

Inside their ``__ua_function__`` implementation, they might need to do something like the
following:

>>> class Backend: pass
>>> meta_backend = Backend()
>>> meta_backend.__ua_domain__ = "numpy"
>>> def ua_func(f, a, kw):
...     # We do this to avoid infinite recursion
...     with ua.skip_backend(meta_backend):
...         # Actual implementation here
...         pass
>>> meta_backend.__ua_function__ = ua_func

In this form, one could do something like the following to use the meta-backend:

>>> with ua.set_backend(DaskBackend(inner=SparseBackend)):
...     x = np.zeros((2000, 2000))
...     isinstance(x, da.Array)
...     isinstance(x.compute(), sparse.SparseArray)
True
True
"""
from ._multimethods import *
from .lib import c_, r_, s_
from . import linalg
from . import lib
from . import random

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
