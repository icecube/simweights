=================================
Why Does Simulation Have Weights?
=================================

This section provides a brief mathematical background on the principle of biased sampling in Monte 
Carlo simulation and the definition of weights in IceCube.

Monte Carlo Integration
=======================

Monte Carlo is a method for calculating the value of definite integrals using random numbers.
Consider the multidimensional integral

.. math::
    
    I = \int_\Omega f(\bar{x}) \mathrm{d}\bar{x}

where :math:`\bar{x}` is an m-dimensional vector and :math:`\Omega` is a subset of :math:`\mathbb{R}^m` with a volume of 

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
===============

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

Weights in IceCube Simulation
=============================

In IceCube, simulation is performed to predict the number of observed events that will be observed assuming
a given flux. The Monte Carlo integral has the following form

.. math::

    N_{obs} = \int \mathrm{d}E \int \mathrm{d}A \int \mathrm{d}\Omega \int \mathrm{d}t 
    \cdot \mathcal{D}(E,\theta,\phi,\vec{x}) \cdot \Phi(E,\theta,\phi,t)

where the ambient flux for a given particle species is given by

.. math::

    \Phi(E,\theta, \phi, t) = \frac{\partial N}{\partial E\ \partial A\ \partial \Omega\ \partial t}

and :math:`\mathcal{D}(E,\theta,\phi,\vec{x})` is a detector response function which describes the amount 
of events which will be actually be detected by IceCube divided by the number of events incident on the
detector volume for a given energy, direction, and impact parameter. 

In principle the flux can vary with time. But, Since IceCube's response to individual particles does not
significantly change over time for the purpose of calculating the detector response, time can be factored 
out of this integral. The quantity we are interested in then becomes the rate of observed events, 
denoted as :math:`R`

.. math::
    
    R = \frac{N_{obs}}{T_{live}} 
    = \int \mathrm{d}E \int \mathrm{d}A \int \mathrm{d}\Omega \cdot \mathcal{D}(E,\theta,\phi,\bar{x}) \cdot \Phi(E,\theta, \phi)

Monte Carlo integration will be performed by sampling primary particles on a 5-dimensional surface of energy, 
direction, and impact parameter on the area perpendicular to the direction of travel. The integral will be

.. math::
    
    R = \sum_{i=1}^{N_{gen}} g_i \cdot \mathcal{D}_i \cdot \Phi(E_i,\theta_i,\phi_i)

The quantity :math:`\mathcal{D}_i` is a binary describing weather or not the :math:`i^{th}` event passed a series of 
quality cuts and :math:`g_i` is generation bias of the :math:`i^{th}` event. Since the majority of simulated events 
won't pass the cut, it is easier from a bookkeeping point of view to simply remove the events from the 
sample that don't pass the quality cuts and only sum over the :math:`N_{pass}` events that do pass the cuts.

.. math::
    
    R = \sum_{i=1}^{N_{pass}} g_i \cdot \Phi(E_i,\theta_i,\phi_i)

It is clear that :math:`g_i` will have units of Energy * Area * Solid Angle, the same units as the surface 
on which primary particles were generated. 

Simulation of IceCube events does not just entail the random sampling of primary particles on a 5 dimensional
surface. It uses random sampling to propagate particles' interaction with matter and the electronics in
the detector. The number of such samplings which occur will often depend on the value of samplings which 
occurred earlier in the particle propagation, making the dimensionality of the Monte Carlo integration vary
from event to event.
In principle the generation bias of an event should be the product of all of the pdfs from every random sampling 
which occurs. However most of these random samplings occur at the natural rate which means that the value 
of the pdf is one and therefore do not have to be kept track of.

Thus the value of the pdf of event :math:`i` is going to be

.. math::
    
    p_i = p_i^{int}\cdot p^{gen}(E_i,\theta_i,\phi_i,\vec{x}_i)

Where :math:`p_i^{int}` is the product the samplings which were not drawn from their natural rate and
:math:`p^{gen}` is the pdf of the 5-dimensional generation surface.
:math:`p_i^{int}` is labeled like this because the most prominent example of such a sampling is the  
the probability of a neutrino interaction as calculated in neutrino-generator.
Since most generators used in IceCube simulation do not bias based on azimuth or impact parameter we will
only show the energy energy and zenith terms.

Assuming that the pdfs for :math:`E` and :math:`\theta` are independent and replacing :math:`p_i^{int}` 
with a generation bias :math:`g_i^{int} = 1 / p_i^{int}` the expression for the generation bias for a single sample becomes

.. math::

    g_i = \frac{g_i^{int}}{N_{gen} \cdot p(E_i) \cdot p(\theta_i)}

:math:`g_i` is essentially the well-known quantity ``OneWeight`` provided by neutrino-generator.
If one is only interested in calculating the weights for a single dataset or multiple datasets which 
use exactly the same pdfs for energy and zenith then an expression like this can be used.
The only care which needs to be taken is that :math:`N_{gen}` is the sum of all generated events across
the combined datasets, which means keeping track of both the number of events per file and the number of
files.

However, if one wishes to combine multiple datasets with different energy power-law indices or 
non-overlapping energy or zenith ranges then more care must be taken.
The generation bias will then become the sum of all of the pdfs for each sample multiplied by the number 
of events generated by each sample. 

.. math::

    g_i =  g_i^{int} \left(\sum_{j=1}^M N_{j} \cdot p_j(E_i) \cdot p_j(\theta_i)\right)^{-1}

Note that :math:`g_i^{int}` is the same for the event regardless of which sample it was generated and 
therefore it can be factored out of the summation. Further note that the pdf for each dataset will be 
evaluated on every event regardless of which dataset it came from. Also, recall that pdfs are defined to be
zero outside of the region on which the dataset generated events.

The quantity that is commonly refered to as the weight in IceCube, :math:`w_i`, is thus defined as 

.. math::
    
    w_i = g_i \cdot \Phi(E_i,\theta_i)

such that 

.. math::

    R = \frac{N_{obs}}{T_{live}} = \sum_i^{N_{pass}} w_i

The purpose of the simweights library is to correctly calculate the values of :math:`g_i` and :math:`w_i`
for combinations of datasets with different generation surfaces.
