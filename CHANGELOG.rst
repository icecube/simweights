.. SPDX-FileCopyrightText: © 2023 the SimWeights contributors
..
.. SPDX-License-Identifier: BSD-2-Clause

Changelog
=========

`v0.1.2`_ - 2023-11-02
----------------------

- add explanation what the effective livetime is -- Ludwig Neste (#17)
- Update the calculation of IceTop weights to fix an issue with the effective area.
  The fix replaces the CircularInjector with NaturalRateCylinder,
  where the cylinder height is set to zero to mimic the flat surface detector.
  The calculated effective area is the projected area, which can be corrected by
  dividing by cos(zenith) -- Mattias Plum & Katherine Rawlins (#22)

`v0.1.1`_ - 2023-07-26
----------------------

- Add Global Spline Fit model -- Ludwig Neste (#10)
- Add support for reading ROOT files with uproot -- Kevin Meagher (418680a)

`v0.1.0`_ - 2023-01-24
----------------------

Initial Release

.. _v0.1.2: https://github.com/icecube/simweights/releases/tag/v0.1.2
.. _v0.1.1: https://github.com/icecube/simweights/releases/tag/v0.1.1
.. _v0.1.0: https://github.com/icecube/simweights/releases/tag/v0.1.0
