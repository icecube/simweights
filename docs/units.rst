.. SPDX-FileCopyrightText: © 2022 the SimWeights contributors
..
.. SPDX-License-Identifier: BSD-2-Clause

Units in SimWeights
===================

Units can be a big source of confusion when weighting simulation because the units most commonly
used for flux are in a different system than those used by offline-software.
IceTray uses a `unit system <https://docs.icecube.aq/icetray/main/projects/icetray/i3units.html>`_
which is convenient for IceCube reconstructions but not for weighting.
The units used in SimWeights were not chosen to be based on a consistent system, but were instead chosen
to be the units most often used for each quantity.
For particle flux per centimeter square because even though for cosmic rays it can be given in either per
centimeter squared or per meter squared, for neutrinos, flux is almost always given per square centimeter.
A table of the units used for each quantity is shown below.

+--------------------+--------------------------------------------------------+
| Length             | :math:`\mathrm{m}`                                     |
+--------------------+--------------------------------------------------------+
| Effective Area     | :math:`\mathrm{m}^2`                                   |
+--------------------+--------------------------------------------------------+
| Solid angle        | :math:`\mathrm{sr}`                                    |
+--------------------+--------------------------------------------------------+
| Etendue            | :math:`\mathrm{cm}^2\cdot\mathrm{sr}`                  |
+--------------------+--------------------------------------------------------+
| Energy             | :math:`\mathrm{GeV}`                                   |
+--------------------+--------------------------------------------------------+
| Generation Surface | :math:`\mathrm{GeV}\cdot\mathrm{cm}^2\cdot\mathrm{sr}` |
+--------------------+--------------------------------------------------------+
| Particle Flux      | :math:`\mathrm{GeV}^{-1}\cdot\mathrm{cm}^{-2}          |
|                    | \cdot\mathrm{sr}^{-1}\cdot\mathrm{s}^{-1}`             |
+--------------------+--------------------------------------------------------+
| Weights            | :math:`\mathrm{s}^{-1}`                                |
+--------------------+--------------------------------------------------------+

.. Note ::

    The units listed here are for the most common case weighing case. The value returned by
    :code:`get_weights()` will be whatever the units passed as the flux times the generation surface.
    For example, if you were to pass a quantity that represented fluence in units of
    :math:`\mathrm{GeV}^{-1}\cdot\mathrm{cm}^{-2}\cdot\mathrm{sr}^{-1}`, then the result would be a
    unitless weight.
