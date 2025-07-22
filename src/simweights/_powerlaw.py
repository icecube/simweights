# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._utils import SeedType, check_random_state

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class PowerLaw:
    r"""A power-law continuous probability distribution.

    This has a similar interface to the probability distribution classes found in :py:mod:`scipy.stats`.
    However, it has several differences needed for weighting Monte Carlo simulation:

    * The support is defined from a to b rather than from 0 to 1.
    * Negative values of the power-law index are allowed.
    * No shape or location parameters are supported.

    The probability density function for a `PowerLaw` is defined as:

    .. math::

        pdf(x, \gamma) = A x^{\gamma}\quad\mathrm{for}\quad a \le x \le b.

    Args:
        g (float): Power-law index
        a (float): Lower bound of the support of the distribution.
        b (float): Upper bound of the support of the distribution.
    """

    # pylint: disable=invalid-name

    def __init__(self: PowerLaw, g: float, a: float, b: float, colname: str | None = None) -> None:
        assert b > a
        self.g = float(g)
        self.a = float(a)
        self.b = float(b)
        self.G = self.g + 1
        if self.G == 0:
            self.integral = np.log(self.b / self.a)
        else:
            self.integral = (self.b**self.G - self.a**self.G) / self.G
        self.span = b - a
        self.columns = (colname,)

    def _pdf(self: PowerLaw, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(x**self.g / self.integral, dtype=np.float64)

    def _cdf(self: PowerLaw, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.G == 0:
            return np.asarray(np.log(x / self.a) / self.integral, dtype=np.float64)
        return np.asarray((x**self.G - self.a**self.G) / self.G / self.integral, dtype=np.float64)

    def _ppf(self: PowerLaw, q: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.G == 0:
            return np.asarray(self.a * np.exp(q * self.integral), dtype=np.float64)
        return np.asarray((q * self.G * self.integral + self.a**self.G) ** (1 / self.G), np.float64)

    def pdf(self: PowerLaw, x: ArrayLike) -> NDArray[np.float64]:
        r"""Probability density function.

        Args:
            x (array_like): quantiles


        Returns:
            array_like: Probability density function evaluated at `x`
        """
        xa = np.asarray(x, dtype=np.float64)
        return np.piecewise(xa, [(xa >= self.a) & (xa <= self.b)], [self._pdf])

    def cdf(self: PowerLaw, x: ArrayLike) -> NDArray[np.float64]:
        r"""Cumulative distribution function.

        Args:
            x (array_like): quantiles

        Returns:
            array_like: Cumulative distribution function evaluated at `x`
        """
        qa = np.asarray(x, dtype=np.float64)
        return np.piecewise(qa, [qa < self.a, qa > self.b], [0, 1, self._cdf])

    def ppf(self: PowerLaw, q: ArrayLike) -> NDArray[np.float64]:
        """Percent point function (inverse of `cdf`) at `q`.

        Args:
            q (array_like): lower tail probability

        Returns:
            array_like: quantile corresponding to the lower tail probability `q`.
        """
        qa = np.asarray(q, dtype=np.float64)
        return np.piecewise(qa, [(qa >= 0) & (qa <= 1)], [self._ppf, np.nan])

    def rvs(self: PowerLaw, size: Any = None, random_state: SeedType = None) -> NDArray[np.float64]:
        """Random variates.

        Args:
            size (int or tuple of ints, optional): Defining number of random variates (Default is 1).
            random_state ({None, int, `~np.random.RandomState`, `~np.random.Generator`}, optional):
                This parameter defines the object to use for drawing random variates.
                If `random_state` is `None` the `~np.random.RandomState` singleton is used.
                If `random_state` is an int, a new ``RandomState`` instance is used, seeded with
                random_state.
                If `random_state` is already a ``RandomState`` or ``Generator`` instance, then that
                object is used.
                Default is None.
        """
        rand_state = check_random_state(random_state)
        return self._ppf(np.asarray(rand_state.uniform(0, 1, size), dtype=np.float64))

    def __repr__(self: PowerLaw) -> str:
        return f"{self.__class__.__name__}({self.g} ,{self.a}, {self.b})"

    def __eq__(self: PowerLaw, other: object) -> bool:
        if not isinstance(other, PowerLaw):
            mesg = f"{self} cannot be compared to {other}"
            raise TypeError(mesg)
        return self.g == other.g and self.a == other.a and self.b == other.b
