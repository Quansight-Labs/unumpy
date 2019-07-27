``unumpy``
==========

.. note::
    This page describes the overall philosophy behind :obj:`unumpy`. If you are
    interested in a general dispatch mechanism, see :obj:`uarray`.

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

There are three main NumPy enhancement proposals (NEPs) inside NumPy itself that relate
to the duck-array ecosystem. There is `NEP-22 <http://www.numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html>`_,
which is a high-level overview of the duck-array ecosystem, and the direction NumPy
intends to move towards. Two main protocols were introduced to fill this gap,
the ``__array_function__`` protocol defined in `NEP-18 <http://www.numpy.org/neps/nep-0018-array-function-protocol.html>`_,
and the older ``__array_ufunc__`` protocol defined in `NEP-13 <https://www.numpy.org/neps/nep-0013-ufunc-overrides.html>`_.

:obj:`unumpy` provides an an alternate framework based on :obj:`uarray`, bypassing
the ``__array_function__`` and ``__array_ufunc__`` protocols entirely. It
provides a clear separation of concerns. It defines callables which can be overridden,
and expresses everything else in terms of these callables. See the :obj:`uarray` documentation
for more details.

For example, you can override ``__call__`` on a ``ufunc`` to override it. You can also override
its methods, ``reduce``, and so on.

Theoretically, you could convert a ``ufunc`` by overriding ``__get__``, but there's also a dedicated
protocol for this.


.. toctree::
    :hidden:
    :maxdepth: 3

    generated/unumpy

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
