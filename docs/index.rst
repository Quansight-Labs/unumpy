``unumpy``
==========

.. note::
    This page describes the overall philosophy behind :obj:`unumpy`. If you are
    interested in an augmentation for NEP-22, please see the :obj:`unumpy` page.

:obj:`unumpy` builds on top of :obj:`uarray`. It is an effort to specify the core
NumPy API, and provide backends for the API.

What's new in ``unumpy``?
-------------------------

:obj:`unumpy` is the first approach to leverage :obj:`uarray` in order to build a
generic backend system for (what we hope will be) the core NumPy API specification.
It's possible to create the backend object, and use that to perform operations.
In addition, it's possible to change the used backend via a context manager.

Relation to the NumPy duck-array ecosystem
------------------------------------------

:obj:`uarray` is a backend/dispatch mechanism with a focus on array computing and the
needs of the wider array community, by allowing a clean way to register an
implementation for any Python object (functions, classes, class methods, properties,
dtypes, ...), it also provides an important building block for
`NEP-22 <http://www.numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html>`_.
It is meant to address the shortcomings of `NEP-18
<http://www.numpy.org/neps/nep-0018-array-function-protocol.html>`_ and `NEP-13
<https://www.numpy.org/neps/nep-0013-ufunc-overrides.html>`_;
while still holding nothing in :obj:`uarray` itself that's specific to array computing
or the NumPy API.

.. toctree::
    :hidden:
    :maxdepth: 3

    generated/unumpy

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
