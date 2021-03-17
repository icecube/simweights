import numpy as np
from scipy._lib._util import check_random_state

class PowerLaw:
    r"""A power-law function continuous random variable.
    
    This has a similar interface to the classes found in :py:module:`scipy.stats` 
    but differs in several ways: support is defined from a to b and negitive 
    values of gamma are allowed. No shape or location parameters are used.

    Notes
    -----
    The probability density function for `PowerLaw` is:
    .. math::
        f(x, \gamma) = A x^{\gamma}
    for :math:`a \le x \le b`.

    Args:
        g (float): Power law index 
        a (float): Lower bound of the support of the distribution.
        b (float): Upper bound of the support of the distribution.
    """
    def __init__(self, g:float, a:float, b:float):
        assert(b > a)
        self.g = float(g)
        self.a = float(a)
        self.b = float(b)
        self.G = self.g + 1
        if self.G == 0:
            self.I = np.log(self.b / self.a)
        else:
            self.I = (self.b**self.G - self.a**self.G) / self.G

        self.span = b - a 

    def _pdf(self, x: float) -> float:
        return x**self.g / self.I

    def _cdf(self, x: float) -> float:
        if self.G==0:
            return np.log(x / self.a) / self.I    
        else:
            return (x**self.G - self.a**self.G) / self.G / self.I

    def _ppf(self, q: float) -> float:
        if self.G == 0:
            return self.a * np.exp(q * self.I)
        else:
            return (q * self.G * self.I + self.a**self.G)**(1 / self.G)

    def pdf(self, x: float) -> float:
        r"""
        Probability density function

        Args:
            x (array_like): quantiles

        Returns:
            array_like: Probability density function evaluated at `x`
        """
        #print('pdf',x,[x<self.a,x>self.b])
        x = np.asfarray(x)
        return np.piecewise(x, [(x >= self.a) & (x <= self.b)], [self._pdf])

    def cdf(self, x: float) -> float:
        r"""
        Cumulative distribution function

        Args:
            x (array_like): quantiles

        Returns:
            array_like: Cumulative distribution function evaluated at `x`
        """
        return np.piecewise(np.asfarray(x), [x < self.a , x > self.b], [0, 1, self._cdf])

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of `cdf`) at `q`.

        Args:
            q (array_like): lower tail probability

        Returns:
            array_like: quantile corresponding to the lower tail probability `q`.
        """
        return np.piecewise(np.asfarray(q),[(q >= 0) & (q <= 1)], [self._ppf, np.nan])


    def rvs(self, size=None, random_state=None):
        """
        Random variates

        Args:
            size (int or tuple of ints, optional): Defining number of random variates (Default is 1).
            random_state ({None, int, `~np.random.RandomState`, `~np.random.Generator`}, optional):
                This parameter defines the object to use for drawing random
                variates.
                If `random_state` is `None` the `~np.random.RandomState` singleton
                is used.
                If `random_state` is an int, a new ``RandomState`` instance is used,
                seeded with random_state.
                If `random_state` is already a ``RandomState`` or ``Generator``
                instance, then that object is used.
                Default is None.
        """
        random_state = check_random_state(random_state)
        return self._ppf(random_state.uniform(0, 1, size))

    def __repr__(self):
        return "{}({:4.2f},{:6.2e},{:6.2e})".format(self.__class__.__name__, self.g, self.a, self.b)

    def __eq__(p1, p2):
        return p1.g == p2.g and p1.a == p2.a and p1.b == p2.b
