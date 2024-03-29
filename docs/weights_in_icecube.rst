.. SPDX-FileCopyrightText: © 2022 the SimWeights contributors
..
.. SPDX-License-Identifier: BSD-2-Clause

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

Monte Carlo integration will be performed by sampling primary particles on a 5-dimensional surface of
energy, direction, and impact parameter on the area perpendicular to the direction of travel.
The integral will be

.. math::

    R = \sum_{i=1}^{N_{gen}} g_i \cdot \mathcal{D}_i \cdot \Phi(E_i,\theta_i,\phi_i)

The quantity :math:`\mathcal{D}_i` is a binary describing weather or not the :math:`i^{th}` event passed a
series of quality cuts and :math:`g_i` is generation bias of the :math:`i^{th}` event. Since the majority
of simulated events won't pass the cut, it is easier from a bookkeeping point of view to simply remove the
events from the sample that don't pass the quality cuts and only sum over the :math:`N_{pass}` events that
do pass the cuts.

.. math::

    R = \sum_{i=1}^{N_{pass}} g_i \cdot \Phi(E_i,\theta_i,\phi_i)

It is clear that :math:`g_i` will have units of Energy * Area * Solid Angle, the same units as the surface
on which primary particles were generated.

Simulation of IceCube events does not just entail the random sampling of primary particles on a 5
dimensional surface. It uses random sampling to propagate particles' interaction with matter and the
electronics in the detector. The number of such samplings which occur will often depend on the value of
samplings which occurred earlier in the particle propagation, making the dimensionality of the Monte Carlo
integration vary from event to event.
In principle the generation bias of an event should be the product of all of the pdfs from every random
sampling which occurs. However most of these random samplings occur at the natural rate which means that
the value of the pdf is one and therefore do not have to be kept track of.

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
with a generation bias :math:`g_i^{int} = 1 / p_i^{int}` the expression for the generation bias for a
single sample becomes

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

The quantity that is commonly referred to as the weight in IceCube, :math:`w_i`, is thus defined as

.. math::

    w_i = g_i \cdot \Phi(E_i,\theta_i)

such that

.. math::

    R = \frac{N_{obs}}{T_{live}} = \sum_i^{N_{pass}} w_i

The purpose of the simweights library is to correctly calculate the values of :math:`g_i` and :math:`w_i`
for combinations of datasets with different generation surfaces.
