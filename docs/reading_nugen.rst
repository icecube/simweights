.. SPDX-FileCopyrightText: © 2022 the SimWeights contributors
..
.. SPDX-License-Identifier: BSD-2-Clause

Reading Neutrino-Generator Files
================================

Correctly interpreting the weighting information for neutrino-generator can be difficult for a number
of reasons. The icetray modules for Neutrino-Generator have a lot of options for generating primary
neutrinos using a number of different generation surfaces.
The scope of SimWeights is limited to configuration which have been used by simulation production and not
try to attempt compatibility with every single configuration option which is available in
neutrino-generator.
SimWeights has been tested to work with all simulation production datasets
going back to at least datasets 10634 which was produced in 2014. Reading older dataset is possible but
will require additional work.

The primary way to weight neutrino-generator data is with the function
:py:class:`simweights.Weighter` which will inspect the ``I3MCWeightDict`` of the given file and
construct the correct ``GenerationSurface`` and return an instance of :py:class:`simweights.Weighter`.

How SimWeights reads I3MCWeightDict
-----------------------------------

The probability that a given neutrino will interact in the simulation volume was originally called
``TotalInteractionProbabilityWeight`` but was renamed ``TotalWeight`` in version 6 of simulation.
SimWeights will use whichever column is present.

The biggest change in neutrino-generator that affected weighting was the change in the zenith distribution.
In version 3 and before of simulation, the cosine of the zenith angle was drawn from a uniform distribution
and the position of the particle was drawn from a circle perpendicular to the primary's momentum.
In version 4 this was changed to generate events on the surface of a cylinder with an
`ad hoc zenith distribution
<https://docs.icecube.aq/icetray/main/projects/neutrino-generator/weighting.html#zenith-weight>`_
to account for a deficit of vertical neutrinos in certain analyses.
The since neutrino generator includes the zenith weight in ``TotalWeight`` and
``TotalInteractionProbabilityWeight`` simweights takes a shortcut an just uses
:py:class:`simweights.UniformSolidAngleCylinder` for the zenith distribution and lets
the factor in ``TotalWeight`` handle the zenith weighting. When this change was first implemented
the size of the cylinder the events were generated on was hard coded to have a height of 1900 m and a
radius of 950 m, the height and radius of the cylinder were added to ``I3MCWeightsDict`` in V6.

All neutrino-generator datasets have been produced with equal number of neutrinos and antineutrinos.
Newer versions of simulation have save the fraction of the type in ``TypeWeight`` but this is assumed
to be 0.5 in files where it is not present.

Event though neutrino-generator saves the minimum and maximum azimuth angle, for simplicity sake,
SimWeights just assumes events were generated on all :math:`2\pi` azimuth.

.. note::

    If you want to use SimWeights to weight custom produced neutrino-generator data which does not conform
    to the weighting scheme used here, it would probably be easier to call :py:class:`simweights.Weighter()`
    directly rather than attempting to shoehorn it into :py:func:`simweights.NuGenWeighter()`.

The complete list of parameters saved by neutrino-generator in the ``I3MCWeightDict`` are documented
`in the icetray documentation <https://docs.icecube.aq/icetray/main/projects/neutrino-generator/weightdict.html>`_.
The columns used by SimWeights are listed in the table below:

===================================== ================================================
Name                                  Notes
===================================== ================================================
``MinZenith``
``MaxZenith``
``InjectionSurfaceR``                 before V04-01-00,
``CylinderHeight``                    after V06-00-00, if not present 1900 is assumed
``CylinderRadius``                    after V06-00-00, if not present 950 is assumed
``MinEnergyLog``
``MaxEnergyLog``
``PowerLawIndex``
``PrimaryNeutrinoType``
``TypeWeight``                        Assumed to be 0.5 if not found
``PrimaryNeutrinoEnergy``
``PrimaryNeutrinoZenith``
``TotalWeight``                       After V06-00-00
``TotalInteractionProbabilityWeight`` Before V06-00-00, replaced by ``TotalWeight``
===================================== ================================================


Why SimWeights doesn't use OneWeight
------------------------------------

The way most people choose to weight neutrino-generator datasets is to simply multiply the ``OneWeight``
by the flux an divide by the number of files. This works fine when you are limited to a single dataset
and is exactly what ``OneWeight`` was designed to do. The issue is that ``OneWeight`` includes all the
information in a way which is difficult to combine multiple datasets with different generation surfaces.
SimWeights takes a different approach and tries to construct the correct representation of the generation
surface.

.. note::

    If you are happy with your existing setup using ``OneWeight`` to weight neutrino-generator data
    for uniform datasets, then you are under no obligation to switch to SimWeights.
