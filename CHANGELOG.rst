.. SPDX-FileCopyrightText: © 2023 the SimWeights contributors
..
.. SPDX-License-Identifier: BSD-2-Clause

Changelog
=========

main - 2023-10-30
---------------------
Update the calculation of IceTop weights to fix an issue with the effective area.
The fix replaces the CircularInjector with NaturalRateCylinder,
where the cylinder height is set to zero to mimic the flat surface detector.
The calculated effective area is the projected area, which can be corrected by
dividing by cos(zenith).

`0.1.0`_ - 2023-01-24
---------------------

Initial Release


.. _0.1.0: https://github.com/icecube/simweights/releases/tag/v0.1.0
