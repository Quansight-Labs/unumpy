import functools
import operator
from uarray import create_multimethod, mark_as, all_of_type, Dispatchable
import builtins

create_numpy = functools.partial(create_multimethod, domain="numpy.random")

from .._multimethods import (
    ClassOverrideMetaWithConstructor,
    ClassOverrideMetaWithGetAttr,
    ClassOverrideMetaWithConstructorAndGetAttr,
    ndarray,
    _identity_argreplacer,
    _dtype_argreplacer,
    _self_argreplacer,
    _skip_self_argreplacer,
    _first2argreplacer,
    _first3argreplacer,
    mark_dtype,
    mark_non_coercible,
)


@create_numpy(_identity_argreplacer)
def default_rng(seed=None):
    return ()


class RandomState(metaclass=ClassOverrideMetaWithConstructor):
    pass


class BitGenerator(metaclass=ClassOverrideMetaWithGetAttr):
    pass


class SeedSequence(metaclass=ClassOverrideMetaWithConstructorAndGetAttr):
    pass


@create_numpy(_identity_argreplacer)
def rand(*tup):
    return ()


@create_numpy(_identity_argreplacer)
def randn(*tup):
    return ()


def _randint_argreplacer(args, kwargs, dispatchables):
    def replacer(low, high=None, size=None, dtype=int):
        return (
            (dispatchables[0],),
            dict(high=dispatchables[1], size=size, dtype=dispatchables[2]),
        )

    return replacer(*args, **kwargs)


@create_numpy(_randint_argreplacer)
@all_of_type(ndarray)
def randint(low, high=None, size=None, dtype=int):
    return (low, high, mark_dtype(dtype))


@create_numpy(_identity_argreplacer)
def random_integers(low, high=None, size=None):
    return ()


@create_numpy(_identity_argreplacer)
def random_sample(size=None):
    return ()


@create_numpy(_identity_argreplacer)
def random(size=None):
    return ()


@create_numpy(_identity_argreplacer)
def ranf(size=None):
    return ()


@create_numpy(_identity_argreplacer)
def sample(size=None):
    return ()


def _choice_argreplacer(args, kwargs, dispatchables):
    def replacer(a, size=None, replace=True, p=None):
        return (dispatchables[0],), dict(size=size, replace=replace, p=dispatchables[1])

    return replacer(*args, **kwargs)


@create_numpy(_choice_argreplacer)
@all_of_type(ndarray)
def choice(a, size=None, replace=True, p=None):
    return (a, p)


@create_numpy(_identity_argreplacer)
def bytes(length):
    return ()


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def shuffle(x):
    return (x,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def permutation(x):
    return (x,)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def beta(a, b, size=None):
    return (a, b)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def binomial(n, p, size=None):
    return (n, p)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def chisquare(df, size=None):
    return (df,)


@create_numpy(_identity_argreplacer)
def dirichlet(alpha, size=None):
    return ()


def _exponential_argreplacer(args, kwargs, dispatchables):
    def replacer(scale=1.0, size=None):
        return (), dict(scale=dispatchables[0], size=size)

    return replacer(*args, **kwargs)


@create_numpy(_exponential_argreplacer)
@all_of_type(ndarray)
def exponential(scale=1.0, size=None):
    return (scale,)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def f(dfnum, dfden, size=None):
    return (dfnum, dfden)


def _gamma_argreplacer(args, kwargs, dispatchables):
    def replacer(shape, scale=1.0, size=None):
        return (dispatchables[0],), dict(scale=dispatchables[1], size=size)

    return replacer(*args, **kwargs)


@create_numpy(_gamma_argreplacer)
@all_of_type(ndarray)
def gamma(shape, scale=1.0, size=None):
    return (shape, scale)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def geometric(p, size=None):
    return (p,)


def _loc_scale_argreplacer(args, kwargs, dispatchables):
    def replacer(loc=0.0, scale=1.0, size=None):
        return (), dict(loc=dispatchables[0], scale=dispatchables[1], size=size)

    return replacer(*args, **kwargs)


@create_numpy(_loc_scale_argreplacer)
@all_of_type(ndarray)
def gumbel(loc=0.0, scale=1.0, size=None):
    return (loc, scale)


@create_numpy(_first3argreplacer)
@all_of_type(ndarray)
def hypergeometric(ngood, nbad, nsample, size=None):
    return (ngood, nbad, nsample)


@create_numpy(_loc_scale_argreplacer)
@all_of_type(ndarray)
def laplace(loc=0.0, scale=1.0, size=None):
    return (loc, scale)


@create_numpy(_loc_scale_argreplacer)
@all_of_type(ndarray)
def logistic(loc=0.0, scale=1.0, size=None):
    return (loc, scale)


def _lognormal_argreplacer(args, kwargs, dispatchables):
    def replacer(mean=0.0, sigma=1.0, size=None):
        return (), dict(mean=dispatchables[0], sigma=dispatchables[1], size=size)

    return replacer(*args, **kwargs)


@create_numpy(_lognormal_argreplacer)
@all_of_type(ndarray)
def lognormal(mean=0.0, sigma=1.0, size=None):
    return (mean, sigma)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def logseries(p, size=None):
    return (p,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def multinomial(n, pvals, size=None):
    return (n,)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def multivariate_normal(mean, cov, size=None, check_valid="warn", tol=1e-8):
    return (mean, cov)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def negative_binomial(n, p, size=None):
    return (n, p)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def noncentral_chisquare(df, nonc, size=None):
    return (df, nonc)


@create_numpy(_first3argreplacer)
@all_of_type(ndarray)
def noncentral_f(dfnum, dfden, nonc, size=None):
    return (dfnum, dfden, nonc)


@create_numpy(_loc_scale_argreplacer)
@all_of_type(ndarray)
def normal(loc=0.0, scale=1.0, size=None):
    return (loc, scale)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def pareto(a, size=None):
    return (a,)


def _poisson_argreplacer(args, kwargs, dispatchables):
    def replacer(lam=1.0, size=None):
        return (), dict(lam=dispatchables[0], size=size)

    return replacer(*args, **kwargs)


@create_numpy(_poisson_argreplacer)
@all_of_type(ndarray)
def poisson(lam=1.0, size=None):
    return (lam,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def power(a, size=None):
    return (a,)


def _rayleigh_argreplacer(args, kwargs, dispatchables):
    def replacer(scale=1.0, size=None):
        return (), dict(scale=dispatchables[0], size=size)

    return replacer(*args, **kwargs)


@create_numpy(_rayleigh_argreplacer)
@all_of_type(ndarray)
def rayleigh(scale=1.0, size=None):
    return (scale,)


@create_numpy(_identity_argreplacer)
def standard_cauchy(size=None):
    return ()


@create_numpy(_identity_argreplacer)
def standard_exponential(size=None):
    return ()


@create_numpy(_identity_argreplacer)
def standard_gamma(shape, size=None):
    return ()


@create_numpy(_identity_argreplacer)
def standard_normal(size=None):
    return ()


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def standard_t(df, size=None):
    return (df,)


@create_numpy(_first3argreplacer)
@all_of_type(ndarray)
def triangular(left, mode, right, size=None):
    return (left, mode, right)


def _uniform_argreplacer(args, kwargs, dispatchables):
    def replacer(low=0.0, high=1.0, size=None):
        return (), dict(low=dispatchables[0], high=dispatchables[1], size=size)

    return replacer(*args, **kwargs)


@create_numpy(_uniform_argreplacer)
@all_of_type(ndarray)
def uniform(low=0.0, high=1.0, size=None):
    return (low, high)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def vonmises(mu, kappa, size=None):
    return (mu, kappa)


@create_numpy(_first2argreplacer)
@all_of_type(ndarray)
def wald(mean, scale, size=None):
    return (mean, scale)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def weibull(a, size=None):
    return (a,)


@create_numpy(_self_argreplacer)
@all_of_type(ndarray)
def zipf(a, size=None):
    return (a,)


@create_numpy(_identity_argreplacer)
def seed(seed=None):
    return ()


@create_numpy(_identity_argreplacer)
def get_state():
    return ()


@create_numpy(_identity_argreplacer)
def set_state(state):
    return ()


def _integers_argreplacer(args, kwargs, dispatchables):
    def replacer(self, low, high=None, size=None, dtype=int, endpoint=False):
        return (
            (self, dispatchables[0],),
            dict(
                high=dispatchables[1],
                size=size,
                dtype=dispatchables[2],
                endpoint=endpoint,
            ),
        )

    return replacer(*args, **kwargs)


def _Generator_choice_argreplacer(args, kwargs, dispatchables):
    def replacer(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        return (
            (self, dispatchables[0],),
            dict(
                size=size,
                replace=replace,
                p=dispatchables[1],
                axis=axis,
                shuffle=shuffle,
            ),
        )

    return replacer(*args, **kwargs)


def _Generator_exponential_argreplacer(args, kwargs, dispatchables):
    def replacer(self, scale=1.0, size=None):
        return (self,), dict(scale=dispatchables[0], size=size)

    return replacer(*args, **kwargs)


def _Generator_gamma_argreplacer(args, kwargs, dispatchables):
    def replacer(self, shape, scale=1.0, size=None):
        return (self, dispatchables[0],), dict(scale=dispatchables[1], size=size)

    return replacer(*args, **kwargs)


def _Generator_loc_scale_argreplacer(args, kwargs, dispatchables):
    def replacer(self, loc=0.0, scale=1.0, size=None):
        return (self,), dict(loc=dispatchables[0], scale=dispatchables[1], size=size)

    return replacer(*args, **kwargs)


def _Generator_lognormal_argreplacer(args, kwargs, dispatchables):
    def replacer(self, mean=0.0, sigma=1.0, size=None):
        return (self,), dict(mean=dispatchables[0], sigma=dispatchables[1], size=size)

    return replacer(*args, **kwargs)


def _Generator_multinomial_argreplacer(args, kwargs, dispatchables):
    def replacer(self, n, pvals, size=None):
        return (self, dispatchables[0], pvals), dict(size=size)

    return replacer(*args, **kwargs)


def _Generator_poisson_argreplacer(args, kwargs, dispatchables):
    def replacer(self, lam=1.0, size=None):
        return (self,), dict(lam=dispatchables[0], size=size)

    return replacer(*args, **kwargs)


def _Generator_rayleigh_argreplacer(args, kwargs, dispatchables):
    def replacer(self, scale=1.0, size=None):
        return (self,), dict(scale=dispatchables[0], size=size)

    return replacer(*args, **kwargs)


def _Generator_uniform_argreplacer(args, kwargs, dispatchables):
    def replacer(self, low=0.0, high=1.0, size=None):
        return (self,), dict(low=dispatchables[0], high=dispatchables[1], size=size)

    return replacer(*args, **kwargs)


class Generator(metaclass=ClassOverrideMetaWithConstructorAndGetAttr):
    @create_numpy(_integers_argreplacer)
    @all_of_type(ndarray)
    def integers(self, low, high=None, size=None, dtype=int, endpoint=False):
        return (low, high, mark_dtype(dtype))

    @create_numpy(_dtype_argreplacer)
    def random(self, size=None, dtype=float, out=None):
        return (mark_dtype(dtype), mark_non_coercible(out))

    @create_numpy(_Generator_choice_argreplacer)
    @all_of_type(ndarray)
    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        return (a, p)

    @create_numpy(_identity_argreplacer)
    def bytes(self, length):
        return ()

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def shuffle(self, x, axis=0):
        return (x,)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def permutation(self, x, axis=0):
        return (x,)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def beta(self, a, b, size=None):
        return (a, b)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def binomial(self, n, p, size=None):
        return (n, p)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def chisquare(self, df, size=None):
        return (df,)

    @create_numpy(_identity_argreplacer)
    def dirichlet(self, alpha, size=None):
        return ()

    @create_numpy(_Generator_exponential_argreplacer)
    @all_of_type(ndarray)
    def exponential(self, scale=1.0, size=None):
        return (scale,)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def f(self, dfnum, dfden, size=None):
        return (dfnum, dfden)

    @create_numpy(_Generator_gamma_argreplacer)
    @all_of_type(ndarray)
    def gamma(self, shape, scale=1.0, size=None):
        return (shape, scale)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def geometric(self, p, size=None):
        return (p,)

    @create_numpy(_Generator_loc_scale_argreplacer)
    @all_of_type(ndarray)
    def gumbel(self, loc=0.0, scale=1.0, size=None):
        return (loc, scale)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def hypergeometric(self, ngood, nbad, nsample, size=None):
        return (ngood, nbad, nsample)

    @create_numpy(_Generator_loc_scale_argreplacer)
    @all_of_type(ndarray)
    def laplace(self, loc=0.0, scale=1.0, size=None):
        return (loc, scale)

    @create_numpy(_Generator_loc_scale_argreplacer)
    @all_of_type(ndarray)
    def logistic(self, loc=0.0, scale=1.0, size=None):
        return (loc, scale)

    @create_numpy(_Generator_lognormal_argreplacer)
    @all_of_type(ndarray)
    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        return (mean, sigma)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def logseries(self, p, size=None):
        return (p,)

    @create_numpy(_Generator_multinomial_argreplacer)
    @all_of_type(ndarray)
    def multinomial(self, n, pvals, size=None):
        return (n,)

    @create_numpy(_identity_argreplacer)
    def multivariate_hypergeometric(
        self, colors, nsample, size=None, method="marginals"
    ):
        return ()

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def multivariate_normal(self, mean, cov, size=None, check_valid="warn", tol=1e-8):
        return (mean, cov)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def negative_binomial(self, n, p, size=None):
        return (n, p)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def noncentral_chisquare(self, df, nonc, size=None):
        return (df, nonc)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        return (dfnum, dfden, nonc)

    @create_numpy(_Generator_loc_scale_argreplacer)
    @all_of_type(ndarray)
    def normal(self, loc=0.0, scale=1.0, size=None):
        return (loc, scale)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def pareto(self, a, size=None):
        return (a,)

    @create_numpy(_Generator_poisson_argreplacer)
    @all_of_type(ndarray)
    def poisson(self, lam=1.0, size=None):
        return (lam,)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def power(self, a, size=None):
        return (a,)

    @create_numpy(_Generator_rayleigh_argreplacer)
    @all_of_type(ndarray)
    def rayleigh(self, scale=1.0, size=None):
        return (scale,)

    @create_numpy(_identity_argreplacer)
    def standard_cauchy(self, size=None):
        return ()

    @create_numpy(_dtype_argreplacer)
    def standard_exponential(self, size=None, dtype=float, method="zig", out=None):
        return (mark_dtype(dtype), mark_non_coercible(out))

    @create_numpy(_dtype_argreplacer)
    def standard_gamma(self, shape, size=None, dtype=float, out=None):
        return (mark_dtype(dtype), mark_non_coercible(out))

    @create_numpy(_dtype_argreplacer)
    def standard_normal(self, size=None, dtype=float, out=None):
        return (mark_dtype(dtype), mark_non_coercible(out))

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def standard_t(self, df, size=None):
        return (df,)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def triangular(self, left, mode, right, size=None):
        return (left, mode, right)

    @create_numpy(_Generator_uniform_argreplacer)
    @all_of_type(ndarray)
    def uniform(self, low=0.0, high=1.0, size=None):
        return (low, high)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def vonmises(self, mu, kappa, size=None):
        return (mu, kappa)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def wald(self, mean, scale, size=None):
        return (mean, scale)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def weibull(self, a, size=None):
        return (a,)

    @create_numpy(_skip_self_argreplacer)
    @all_of_type(ndarray)
    def zipf(self, a, size=None):
        return (a,)
