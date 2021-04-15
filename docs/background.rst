==========
Background
==========

This section provides a brief background on the principle of Monte Carlo simulation and the 
calculation of weights. 

Monte Carlo Integration
=======================

Monte Carlo is a method for calculating the value of definite integrals using random numbers.
Consider the multidimensional integral

.. math::
    
    I = \int_\Omega f(\vec{x}) \mathrm{d}\vec{x}

where :math:`\vec{x}` is an m-dimensional vector and :math:`\Omega` is a subset of :math:`\mathbb{R}^m` with a volume of 

.. math::
    
    V = \int_\Omega \mathrm{d}\vec{x}

The integral can be approximated by the statement

.. math::

    I \approx \frac{V}{N} \sum_{i=1}^N f(\vec{x}_i)

where :math:`\vec{x}_1, ..., \vec{x}_N` is sampled from :math:`\Omega`. 
If :math:`\vec{x}_i` are sampled from a grid evenly spaced across :math:`\Omega` then this is a Riemann sum.
If :math:`\vec{x}_i` are sampled randomly from :math:`\Omega` then this is Monte Carlo integrations. 
In general Riemann sums are more efficient for 1 dimension and Monte Carlo is more efficient for higher 
dimensions.

Biased Sampling
===============

There is no reason that the values of \vec{x}_i need to be sampled from a uniform distribution.
It is often advantageous to sample from some other distribution denoted :math:`p(\vec{x_i})`.
In this case the integral becomes

.. math::

    I \approx \frac{1}{N} \sum_{i=1}^{N} \frac{f(\vec{x}_i)}{p(\vec{x}_i)}

Note that if :math:`p` is the uniform distribution then :math:`p(\vec{x}_i) = 1 / V` which simplifies 
to the statement above for I.

If two different samples were produced with :math:`N_1` events from :math:`p_1` and :math:`N_2` events 
from :math:`p_2` then the total pdf for the combined sample is

.. math::

    p(\vec{x}_i) = \frac{N_1 \cdot p_1(\vec{x}_i) + N_2 \cdot p_2(\vec{x}_i)}{N_1 + N_2}

plugging this into the summation results in 

.. math::

    I \approx \sum_{i=1}^{N_1+N_2} \frac{f(\vec{x}_i)}{N_1 \cdot p_1(\vec{x}_i) + N_2 \cdot p_2(\vec{x}_i)}


This is further simplified by introducing the quantity called the weight :math:`w(\vec{x}_i)` such that

.. math:: 

    I \approx \sum_{i=1}^{N} w(\vec{x}_i) \cdot P(\vec{x}_i)

where the weight generalized to M samples is defined as

.. math::

    w(\vec{x}_i) = \left({\sum_{j=1}^M N_j \cdot p_j(\vec{x}_i)}\right)^{-1}

Note that this result holds even for samples drawn from disjoint :math:`\Omega`\ s. As long as 
:math:`p_j(\vec{x_i}) = 0` for events outside of :math:`\Omega_j` then the integral returns
the result for the union of all the :math:`\Omega_j`\ s.

IceCube Simulation
==================

In IceCube the integral to be performed by simulation is calculating the number of observed events

.. math::

    N_{obs} = \int \mathrm{d}E \int \mathrm{d}A \int \mathrm{d}\Omega \int \mathrm{d}t 
    \cdot D(E,\theta,\phi,\vec{x}) \cdot \Phi(E,\theta,\phi,t)

where 

.. math::

    \Phi(E,\theta, \phi t) = \frac{\partial N}{\partial E\ \partial A\ \partial \Omega\ \partial t}

is the ambient flux of a certain particle species and :math:`D(E,\theta,\phi,\vec{x})` is a
detector response function which describes weather not a given particle will be detected by the detector.

In principle the flux can be time varying but that is usually neglected so that the time component can be factored out and 
the quantity we are interested is expressed as a rate :math:`R`

.. math::
    
    R = \frac{N_{obs}}{T} 
    = \int \mathrm{d}E \int \mathrm{d}A \int \mathrm{d}\Omega \cdot D(E,\theta,\phi,\vec{x}) \cdot \Phi(E,\theta)

Monte Carlo integration will be performed by sampling primary particles on a 5-dimensional surface of energy, 
direction, and area perpendicular to the direction of travel. The integral will be

.. math::
    
    R = \sum_{i=1}^{N_{gen}} w_i \cdot D_i \cdot \Phi(E_i,\theta_i,\phi_i)

The quantity :math:`D_i` is a binary describing weather or not the :math:`i^{th}` event passed a series of 
quality cuts and :math:`w_i` is weight of the :math:`i^{th}` event. Since the majority of simulated events 
won't pass the cut, it is easier from a book keeping point of view to simply remove the events from the 
sample that don't pass the cut and only sum over the :math:`N_{pass}` events that do pass the cuts.

.. math::
    
    R = \sum_{i=1}^{N_{pass}} w_i \cdot \Phi(E_i,\theta_i,\phi_i)

It is clear that :math:`w_i` will have units of Energy * Area * Solid Angle, the same units as the surface 
that the primary particles were generated. 

The pdf of an event will be the product of the pdf for each sampling which occurs. 
Note that the 5-dimensional surface on which the primary particle is sampled from is not the only sampling 
which occurs during simulation. Simulation of the propagation of secondary particles is also performed 
using the Monte Carlo method. In principle the weight should take into account all of these process. 
However, most of these are sampled at the rate they occur in nature that is a number is sampled from the 
uniform distribution :math:`U(0,1)` and plugged into the cdf of the processes in question this results in 
a pdf value of one. Thus the probability for event :math:`i` is

.. math::
    
    p_i = p_i^{int}\cdot p^{gen}(E_i,\theta_i)

Where :math:`p^{int}` is the product of the non-uniform pdfs for this event.
This is called this because the most prominent example of such a sampling is the  
the interaction probability in neutrino-generator.
In addition, since in most cases the generation probability is only biased in energy and zenith
so we will drop the other terms.

The Weight is then for a single sample where the pdfs for :math:`E` and :math:`\theta` are independent is then

.. math::

    w_i = \frac{w_i^{int}}{N_{gen} \cdot p(E_i) \cdot p(\theta_i)}

This is essentially the well known quantity referred to as ``OneWeight`` provided by neutrino-generator.

However, if multiple datasets are combined we need to account for the different generation surfaces.
The weight then

.. math::

    w_i =  w_i^{int} \left(\sum_{j=1}^M N_{j} \cdot p_j(E_i) \cdot p_j(\theta_i)\right)^{-1}

Note that :math:`w_i^{int}` is the same for the event regardless of which sample it was generated and 
therefore it can be factored out of the summation.

