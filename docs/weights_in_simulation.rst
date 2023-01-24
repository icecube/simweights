.. SPDX-FileCopyrightText: © 2022 the SimWeights contributors
..
.. SPDX-License-Identifier: BSD-2-Clause

Weighting in Monte Carlo Simulation
===================================

This page provides an introduction to the principle of biased sampling in Monte
Carlo simulation and the definition of weights in IceCube.

Monte Carlo Integration
-----------------------

Monte Carlo is a method for calculating the value of definite integrals using random numbers.
Consider the multidimensional integral

.. math::

    I = \int_\Omega f(\bar{x}) \mathrm{d}\bar{x}

where :math:`\bar{x}` is an m-dimensional vector and :math:`\Omega` is a subset of :math:`\mathbb{R}^m`
with a volume of

.. math::

    V = \int_\Omega \mathrm{d}\bar{x}

The integral can be approximated by the statement

.. math::

    I \approx \frac{V}{N} \sum_{i=1}^N f(\bar{x}_i)

where :math:`\bar{x}_1, ..., \bar{x}_N` is sampled from :math:`\Omega`.
If :math:`\bar{x}_i` are points sampled from a grid evenly spaced across :math:`\Omega` then this is known
as a Riemann sum. However, if :math:`\bar{x}_i` are points randomly sampled from :math:`\Omega` then this
is known as Monte Carlo integration. In general Riemann sums are more efficient for 1 dimension and Monte
Carlo is more efficient at higher dimensions.

Biased Sampling
---------------

There is no reason that the values of :math:`\bar{x}_i` need to be sampled from a uniform distribution
on :math:`\Omega`. It is often advantageous to sample from some other probability distribution function,
which is denoted by :math:`p(\bar{x_i})`. In this case the integral becomes

.. math::

    I \approx \frac{1}{N} \sum_{i=1}^{N} \frac{f(\bar{x}_i)}{p(\bar{x}_i)}

Note that if :math:`p` is the uniform distribution then :math:`p(\bar{x}_i) = 1 / V` which simplifies
to the statement above for :math:`I`.

If two samples were produced from two different distributions: :math:`N_1` events drawn from :math:`p_1`
and :math:`N_2` events drawn from :math:`p_2` then the total pdf for the combined sample
:math:`\bar{x}_1, ..., \bar{x}_{N_1+N_2}` becomes

.. math::

    p(\bar{x}_i) = \frac{N_1 \cdot p_1(\bar{x}_i) + N_2 \cdot p_2(\bar{x}_i)}{N_1 + N_2}

So that the Monte Carlo integral statement becomes

.. math::

    I \approx \sum_{i=1}^{N_1+N_2} \frac{f(\bar{x}_i)}{N_1 \cdot p_1(\bar{x}_i) + N_2 \cdot p_2(\bar{x}_i)}


To make things easier to to keep track of we can introduce a quantity called the generation bias
:math:`g(\bar{x}_i)` such that

.. math::

    I \approx \sum_{i=1}^{N} g(\bar{x}_i) \cdot f(\bar{x}_i)

where the generation bias generalized to M samples is defined as

.. math::

    g(\bar{x}_i) = \left({\sum_{j=1}^M N_j \cdot p_j(\bar{x}_i)}\right)^{-1}

Note that this result holds even for samples drawn from disjoint surfaces. As long as the pdf for the
:math:`j^{th}` sample is defined such that :math:`p_j(\bar{x_i}) = 0` for events outside of
:math:`\Omega_j`, then the Monte Carlo integral will give the correct answer for integration on the surface
of the union of all the :math:`\Omega_j`.
