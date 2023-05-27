# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause
# mypy: disable-error-code="no-any-return"

"""
A collection of cosmic ray flux parametrizations.

This files contains the same Cosmic Ray flux models as :file:`weighting/python/fluxes.py`.
However they have been refactored to:

* output units in cm^2 rather than m^2
* Use PDG particle codes natively instead of CORSIKA code
* Use :py:func:`numpy.piecewise` instead of :py:mod:`numexpr`
* Follow :py:mod:`numpy` broadcasting rules

"""


from typing import Callable, List, Mapping, Optional, Union
from pathlib import Path

from numpy import asfarray, bool_, broadcast_arrays, exp, float64, int32, piecewise, sqrt, genfromtxt
from numpy import sum as nsum
from numpy.typing import ArrayLike, NDArray

from scipy.interpolate import CubicSpline  # pylint: disable=import-error

from ._pdgcode import PDGCode

# pylint: disable=too-few-public-methods
# flake8: noqa: N803


class CosmicRayFlux:
    """
    Base class for cosmic ray fluxes that uses :py:func:`numpy.piecewise` for efficient
    mathematical evaluation

    Derived must set `pdgids` to enumerate the particle types in this model.
    :py:func:`_funcs` must be set to a list of functions to be called for each particle
    type. :py:func:`_condition()` can be overridden if additional piecewise conditions are
    desired.
    """

    pdgids: List[PDGCode] = []
    _funcs: List[Union[float, Callable[[float], float]]] = []

    def _condition(
        self,
        energy: NDArray[float64],
        pdgid: NDArray[int32],
    ) -> List[NDArray[bool_]]:
        # pylint: disable=unused-argument
        return [pdgid == p for p in self.pdgids]

    def __call__(self, energy: ArrayLike, pdgid: ArrayLike) -> NDArray[float64]:
        energy_arr, pdgid_arr = broadcast_arrays(energy, pdgid)
        pcond = self._condition(energy_arr, pdgid_arr)
        return piecewise(energy, pcond, self._funcs)


class Hoerandel(CosmicRayFlux):
    r"""
    All-particle spectrum (up to iron) after Hörandel\ [#Hoerandel]_, as implemented
    in dCORSIKA.
    """

    pdgids = [
        PDGCode.PPlus,
        PDGCode.He4Nucleus,
        PDGCode.Li7Nucleus,
        PDGCode.Be9Nucleus,
        PDGCode.B11Nucleus,
        PDGCode.C12Nucleus,
        PDGCode.N14Nucleus,
        PDGCode.O16Nucleus,
        PDGCode.F19Nucleus,
        PDGCode.Ne20Nucleus,
        PDGCode.Na23Nucleus,
        PDGCode.Mg24Nucleus,
        PDGCode.Al27Nucleus,
        PDGCode.Si28Nucleus,
        PDGCode.P31Nucleus,
        PDGCode.S32Nucleus,
        PDGCode.Cl35Nucleus,
        PDGCode.Ar40Nucleus,
        PDGCode.K39Nucleus,
        PDGCode.Ca40Nucleus,
        PDGCode.Sc45Nucleus,
        PDGCode.Ti48Nucleus,
        PDGCode.V51Nucleus,
        PDGCode.Cr52Nucleus,
        PDGCode.Mn55Nucleus,
        PDGCode.Fe56Nucleus,
    ]
    _funcs = [
        lambda E: 1.1776445965025136 * E**-2.71 * (1 + (E / (4.49e6 * 1)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.4749371132996256 * E**-2.64 * (1 + (E / (4.49e6 * 2)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.00867088317618298 * E**-2.54 * (1 + (E / (4.49e6 * 3)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.008429044403584496 * E**-2.75 * (1 + (E / (4.49e6 * 4)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.06336114770238042 * E**-2.95 * (1 + (E / (4.49e6 * 5)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.10122921411827233 * E**-2.66 * (1 + (E / (4.49e6 * 6)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.03396783461252935 * E**-2.72 * (1 + (E / (4.49e6 * 7)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.17214707679448027 * E**-2.68 * (1 + (E / (4.49e6 * 8)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.0038536639802016566 * E**-2.69 * (1 + (E / (4.49e6 * 9)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.038261133470722905 * E**-2.64 * (1 + (E / (4.49e6 * 10)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.007200644098601636 * E**-2.66 * (1 + (E / (4.49e6 * 11)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.06662427806532402 * E**-2.64 * (1 + (E / (4.49e6 * 12)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.010982414739246525 * E**-2.66 * (1 + (E / (4.49e6 * 13)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.14155104103909828 * E**-2.75 * (1 + (E / (4.49e6 * 14)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.0031722233983367293 * E**-2.69 * (1 + (E / (4.49e6 * 15)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.010229054260257044 * E**-2.55 * (1 + (E / (4.49e6 * 16)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.003223645896660968 * E**-2.68 * (1 + (E / (4.49e6 * 17)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.006953545126418338 * E**-2.64 * (1 + (E / (4.49e6 * 18)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.004777105028396875 * E**-2.65 * (1 + (E / (4.49e6 * 19)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.018506203553374286 * E**-2.70 * (1 + (E / (4.49e6 * 20)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.002528561864152123 * E**-2.64 * (1 + (E / (4.49e6 * 21)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.007639737621929389 * E**-2.61 * (1 + (E / (4.49e6 * 22)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.0048981193059270424 * E**-2.63 * (1 + (E / (4.49e6 * 23)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.013916784695018254 * E**-2.67 * (1 + (E / (4.49e6 * 24)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.0032384244406763116 * E**-2.46 * (1 + (E / (4.49e6 * 25)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.12012410569254007 * E**-2.59 * (1 + (E / (4.49e6 * 26)) ** 1.9) ** (-2.1 / 1.9),
    ]


class Hoerandel5(CosmicRayFlux):
    r"""
    Hoerandel with only 5 components, after Becherini et al.\ [#Becherini]_
    (These are the same as used by Arne Schoenwald's version\ [#Schoenwald]_)
    """
    pdgids = [
        PDGCode.PPlus,
        PDGCode.He4Nucleus,
        PDGCode.N14Nucleus,
        PDGCode.Al27Nucleus,
        PDGCode.Fe56Nucleus,
    ]
    _funcs = [
        lambda E: 1.1776445965025136 * E**-2.71 * (1 + (E / (4.49e6 * 1)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.4749371132996256 * E**-2.64 * (1 + (E / (4.49e6 * 2)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.35525893555039243 * E**-2.68 * (1 + (E / (4.49e6 * 7)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.32336058556071825 * E**-2.67 * (1 + (E / (4.49e6 * 13)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.11979991050096223 * E**-2.58 * (1 + (E / (4.49e6 * 26)) ** 1.9) ** (-2.1 / 1.9),
    ]


class Hoerandel_IT(CosmicRayFlux):
    """
    Modified 5-component Hoerandel spectrum with N and Al replaced by O.
    """

    # pylint: disable=invalid-name
    pdgids = [PDGCode.PPlus, PDGCode.He4Nucleus, PDGCode.O16Nucleus, PDGCode.Fe56Nucleus]
    _funcs = [
        lambda E: 1.1776445965025136 * E**-2.71 * (1 + (E / (4.49e6 * 1)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.4749371132996256 * E**-2.64 * (1 + (E / (4.49e6 * 2)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.7017460455316394 * E**-2.68 * (1 + (E / (4.49e6 * 8)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 0.11979991050096223 * E**-2.58 * (1 + (E / (4.49e6 * 26)) ** 1.9) ** (-2.1 / 1.9),
    ]


class GaisserHillas(CosmicRayFlux):
    r"""
    Spectral fits from an internal report\ [#Gaisser1]_ and in Astropart. Phys\ [#Gaisser2]_ by Tom Gaisser.
    """
    pdgids = [
        PDGCode.PPlus,
        PDGCode.He4Nucleus,
        PDGCode.N14Nucleus,
        PDGCode.Al27Nucleus,
        PDGCode.Fe56Nucleus,
    ]
    _funcs = [
        lambda E: 0.7860 * E ** (-2.66) * exp(-E / (4e6 * 1)),
        lambda E: 0.3550 * E ** (-2.58) * exp(-E / (4e6 * 2)),
        lambda E: 0.2200 * E ** (-2.63) * exp(-E / (4e6 * 7)),
        lambda E: 0.1430 * E ** (-2.67) * exp(-E / (4e6 * 13)),
        lambda E: 0.2120 * E ** (-2.63) * exp(-E / (4e6 * 26)),
    ]


class GaisserH3a(CosmicRayFlux):
    r"""
    Spectral fits from an internal report\ [#Gaisser1]_ and in Astropart. Phys\ [#Gaisser2]_ by Tom Gaisser.

    The model H3a with a mixed extra-galactic population (Fig. 2)
    has all iron at the highest energy and would represent a
    scenario in which the cutoff is not an effect of energy loss
    in propagation over cosmic distances through the CMB but is
    instead just the extra-galactic accelerators reaching their
    highest energy.
    """
    pdgids = [
        PDGCode.PPlus,
        PDGCode.He4Nucleus,
        PDGCode.N14Nucleus,
        PDGCode.Al27Nucleus,
        PDGCode.Fe56Nucleus,
    ]
    _funcs = [
        lambda E: 0.7860 * E**-2.66 * exp(-E / (4e6 * 1))
        + 0.0020 * E**-2.4 * exp(-E / (3e7 * 1))
        + 0.00017 * E**-2.4 * exp(-E / (2e9 * 1)),
        lambda E: 0.3550 * E**-2.58 * exp(-E / (4e6 * 2))
        + 0.0020 * E**-2.4 * exp(-E / (3e7 * 2))
        + 0.00017 * E**-2.4 * exp(-E / (2e9 * 2)),
        lambda E: 0.2200 * E**-2.63 * exp(-E / (4e6 * 7))
        + 0.00134 * E**-2.4 * exp(-E / (3e7 * 7))
        + 0.000114 * E**-2.4 * exp(-E / (2e9 * 7)),
        lambda E: 0.1430 * E**-2.67 * exp(-E / (4e6 * 13))
        + 0.00134 * E**-2.4 * exp(-E / (3e7 * 13))
        + 0.000114 * E**-2.4 * exp(-E / (2e9 * 13)),
        lambda E: 0.2120 * E**-2.63 * exp(-E / (4e6 * 26))
        + 0.00134 * E**-2.4 * exp(-E / (3e7 * 26))
        + 0.000114 * E**-2.4 * exp(-E / (2e9 * 26)),
    ]


class GaisserH4a(CosmicRayFlux):
    r"""
    Spectral fits from an internal report\ [#Gaisser1]_ and in Astropart. Phys\ [#Gaisser2]_ by Tom Gaisser.

    In the model H4a, on the other hand, the extra-galactic component
    is assumed to be all protons.
    """
    pdgids = [
        PDGCode.PPlus,
        PDGCode.He4Nucleus,
        PDGCode.N14Nucleus,
        PDGCode.Al27Nucleus,
        PDGCode.Fe56Nucleus,
    ]
    _funcs = [
        lambda E: 0.7860 * E**-2.66 * exp(-E / (4e6 * 1))
        + 0.0020 * E**-2.4 * exp(-E / (3e7 * 1))
        + 0.0200 * E**-2.6 * exp(-E / 6e10),
        lambda E: 0.3550 * E**-2.58 * exp(-E / (4e6 * 2)) + 0.002 * E**-2.4 * exp(-E / (3e7 * 2)),
        lambda E: 0.2200 * E**-2.63 * exp(-E / (4e6 * 7)) + 0.00134 * E**-2.4 * exp(-E / (3e7 * 7)),
        lambda E: 0.1430 * E**-2.67 * exp(-E / (4e6 * 13)) + 0.00134 * E**-2.4 * exp(-E / (3e7 * 13)),
        lambda E: 0.2120 * E**-2.63 * exp(-E / (4e6 * 26)) + 0.00134 * E**-2.4 * exp(-E / (3e7 * 26)),
    ]


class GaisserH4a_IT(CosmicRayFlux):
    r"""
    Variation of Gaisser's H4a flux using only four components.

    *This is not a very physical flux*: The Oxygen group is the sum of H4a's Nitrogen and Aluminum groups.
    This is the flux used as an "a priori" estimate of mass-composition to produce the IceTop-only
    flux\ [#Aartsen]_.
    """
    # pylint: disable=invalid-name
    pdgids = [PDGCode.PPlus, PDGCode.He4Nucleus, PDGCode.O16Nucleus, PDGCode.Fe56Nucleus]
    _funcs = [
        lambda E: 0.7860 * E**-2.66 * exp(-E / (4e6 * 1))
        + 0.0020 * E**-2.4 * exp(-E / (3e7 * 1))
        + 0.0200 * E**-2.6 * exp(-E / 6e10),
        lambda E: 0.3550 * E**-2.58 * exp(-E / (4e6 * 2)) + 0.0020 * E**-2.4 * exp(-E / (3e7 * 2)),
        lambda E: 0.2200 * E**-2.63 * exp(-E / (4e6 * 7))
        + 0.00134 * E**-2.4 * exp(-E / (3e7 * 7))
        + 0.1430 * E**-2.67 * exp(-E / (4e6 * 13))
        + 0.00134 * E**-2.4 * exp(-E / (3e7 * 13)),
        lambda E: 0.2120 * E**-2.63 * exp(-E / (4e6 * 26)) + 0.00134 * E**-2.4 * exp(-E / (3e7 * 26)),
    ]


class Honda2004(CosmicRayFlux):
    r"""
    Spectrum used to calculate neutrino fluxes in Honda et al. (2004)\ [#Honda]_.
    (Table 1, with modification from the text).

    Note:
        the E_k notation means energy per nucleon!
    """
    pdgids = [
        PDGCode.PPlus,
        PDGCode.He4Nucleus,
        PDGCode.N14Nucleus,
        PDGCode.Al27Nucleus,
        PDGCode.Fe56Nucleus,
    ]
    _funcs = [
        lambda E: (1.49) * (E + 2.15 * exp(-0.21 * sqrt(E))) ** (-2.74),
        lambda E: (1.49) * (100 ** (2.71 - 2.74)) * (E + 2.15 * exp(-0.21 * sqrt(E))) ** (-2.71),
        lambda E: (0.06 / 4.02) * (E / 4.02 + 1.25 * exp(-0.14 * sqrt(E / 4.02))) ** (-2.64),
        lambda E: (0.00332 / 14.07) * (E / 14.07 + 0.97 * exp(-0.01 * sqrt(E / 14.07))) ** (-2.60),
        lambda E: (0.00342 / 27.13) * (E / 27.13 + 2.14 * exp(-0.01 * sqrt(E / 27.13))) ** (-2.79),
        lambda E: (0.000445 / 56.26) * (E / 56.26 + 3.07 * exp(-0.41 * sqrt(E / 56.26))) ** (-2.68),
    ]

    def _condition(self, energy: NDArray[float64], pdgid: NDArray[int32]) -> List[NDArray[bool_]]:
        energy_break = 100
        return [
            (pdgid == PDGCode.PPlus) * (energy < energy_break),
            (pdgid == PDGCode.PPlus) * (energy >= energy_break),
            pdgid == PDGCode.He4Nucleus,
            pdgid == PDGCode.N14Nucleus,
            pdgid == PDGCode.Al27Nucleus,
            pdgid == PDGCode.Fe56Nucleus,
        ]


class TIG1996(CosmicRayFlux):
    r"""
    Spectrum used to calculate prompt neutrino fluxes in Enberg et al. (2008)\ [#Enberg]_ (Eq. 30).
    The parameterization was taken directly from an earlier paper by Thunman et al\ [#Thunman]_.
    Only the nucleon flux was given, so for simplicity we treat it as a proton-only flux.
    """
    pdgids = [PDGCode.PPlus]
    _funcs = [lambda E: 1.70 * E**-2.7, lambda E: 1.74e2 * E**-3.0, 0]

    def _condition(self, energy: NDArray[float64], pdgid: NDArray[int32]) -> List[NDArray[bool_]]:
        energy_break = 5e6
        return [
            (pdgid == PDGCode.PPlus) * (energy < energy_break),
            (pdgid == PDGCode.PPlus) * (energy >= energy_break),
        ]


class GlobalFitGST(CosmicRayFlux):
    r"""
    Spectral fits by Gaisser, Stanev and Tilav\ [#GaisserStanevTilav]_.
    """
    pdgids = [
        PDGCode.PPlus,
        PDGCode.He4Nucleus,
        PDGCode.N14Nucleus,
        PDGCode.Al27Nucleus,
        PDGCode.Fe56Nucleus,
    ]
    _funcs = [
        lambda E: 0.7 * E**-2.66 * exp(-E / 1.2e5)
        + 0.015 * E**-2.4 * exp(-E / 4e6)
        + 0.0014 * E**-2.4 * exp(-E / 1.3e9),
        lambda E: 0.32 * E**-2.58 * exp(-E / 1.2e5 / 2) + 0.0065 * E**-2.3 * exp(-E / 4e6 / 2),
        lambda E: 0.01 * E**-2.40 * exp(-E / 1.2e5 / 7) + 0.0006 * E**-2.3 * exp(-E / 4e6 / 7),
        lambda E: 0.013 * E**-2.40 * exp(-E / 1.2e5 / 13) + 0.0007 * E**-2.3 * exp(-E / 4e6 / 13),
        lambda E: 0.006 * E**-2.30 * exp(-E / 1.2e5 / 26)
        + 0.00023 * E**-2.2 * exp(-E / 4e6 / 26)
        + 0.0000025 * E**-2.2 * exp(-E / 1.3e9 / 26),
    ]


class GlobalSplineFit(CosmicRayFlux):
    r"""
    Data-driven spline fit of the cosmic ray spectrum by Dembinski et. al. \ [#GSFDembinski]
    """
    pdgids = [
        PDGCode.PPlus,
        PDGCode.He4Nucleus,
        PDGCode.Li7Nucleus,
        PDGCode.Be9Nucleus,
        PDGCode.B11Nucleus,
        PDGCode.C12Nucleus,
        PDGCode.N14Nucleus,
        PDGCode.O16Nucleus,
        PDGCode.F19Nucleus,
        PDGCode.Ne20Nucleus,
        PDGCode.Na23Nucleus,
        PDGCode.Mg24Nucleus,
        PDGCode.Al27Nucleus,
        PDGCode.Si28Nucleus,
        PDGCode.P31Nucleus,
        PDGCode.S32Nucleus,
        PDGCode.Cl35Nucleus,
        PDGCode.Ar40Nucleus,
        PDGCode.K39Nucleus,
        PDGCode.Ca40Nucleus,
        PDGCode.Sc45Nucleus,
        PDGCode.Ti48Nucleus,
        PDGCode.V51Nucleus,
        PDGCode.Cr52Nucleus,
        PDGCode.Mn55Nucleus,
        PDGCode.Fe56Nucleus,
        PDGCode.Co59Nucleus,
        PDGCode.Ni59Nucleus,
    ]

    def __init__(self) -> None:
        data = genfromtxt(Path(__file__).parent / "gsf_data_table.txt")
        energy = data.T[0]
        elements = data.T[1:]
        self._funcs = []
        for element in elements:
            self._funcs.append(CubicSpline(energy, element, extrapolate=False, axis=0))


class GlobalSplineFit5Comp(CosmicRayFlux):
    r"""
    Sum of the flux of the GSF model for the standard 5 components injected by IceCube.
    GSF is a Data-driven spline fit of the cosmic ray spectrum by Dembinski et. al. \ [#GSFDembinski]
    """
    pdgids = [
        PDGCode.PPlus,
        PDGCode.He4Nucleus,
        PDGCode.N14Nucleus,
        PDGCode.Al27Nucleus,
        PDGCode.Fe56Nucleus,
    ]

    groups = [(1, 1), (2, 5), (6, 11), (12, 15), (16, 27)]

    def __init__(self) -> None:
        data = genfromtxt(Path(__file__).parent / "gsf_data_table.txt")
        energy = data.T[0]
        elements = data.T[1:]
        self._funcs = []
        for z_low, z_high in self.groups:
            self._funcs.append(
                CubicSpline(energy, nsum(elements[z_low - 1 : z_high], axis=0), extrapolate=False, axis=0)
            )


class FixedFractionFlux(CosmicRayFlux):
    """
    Total energy per particle flux flux split among mass groups with a constant fraction.

    By default, this takes as basis the flux from Gaisser H4a summed over all its mass groups,
    then multiplies it by the given fraction. This is a quick way to consider different
    weightings for systematic checks.
    """

    def __init__(
        self,
        fractions: Mapping[PDGCode, float],
        basis: Optional[CosmicRayFlux] = None,
        normalized: bool = True,
    ):
        """
        :param fractions: A dictionary of fractions. They must add up to one and they
        should correspond to the pdgids in basis

        :type fractions: a dictionary with dataclasses.ParticleType as keys
        """
        if basis:
            self.flux = basis
        else:
            self.flux = GaisserH4a_IT()
        fluxes = {int(k): 0.0 for k in self.flux.pdgids}
        fluxes.update({int(k): v for k, v in fractions.items()})
        self.pdgids = [PDGCode(k) for k in fluxes]
        self.fracs = list(fluxes.values())
        if normalized:
            assert sum(self.fracs) == 1.0

    def __call__(self, energy: ArrayLike, pdgid: ArrayLike) -> NDArray[float64]:
        """
        :param E: particle energy in GeV
        :param pdgid: particle type code
        :type pdgid: int
        """
        energy_arr, pdgid_arr = broadcast_arrays(energy, pdgid)
        fluxsum = sum(self.flux(energy_arr, p) for p in self.pdgids)
        cond = self._condition(energy_arr, pdgid_arr)
        return asfarray(fluxsum * piecewise(energy, cond, self.fracs))


class _references:
    """
    .. [#Hoerandel] J. R. Hörandel, "On the knee in the energy spectrum of cosmic rays,"
       `Astropart. Phys. 19, 193 (2003)
       <https://doi.org/10.1016/S0927-6505(02)00198-6>`_.
       `astro-ph/0210453 <https://arxiv.org/abs/astro-ph/0210453>`_.
    .. [#Becherini] Y. Becherini et al.,
       "A parameterisation of single and multiple muons in the deep water or ice,"
       `Astropart. Phys. 25, 1 (2006)
       <https://doi.org/10.1016/j.astropartphys.2005.10.005>`_.
       `hep-ph/0507228 <https://arxiv.org/abs/hep-ph/0507228>`_.
    .. [#Schoenwald] A. Schoenwald, ``http://www.ifh.de/~arnes/Forever/Hoerandel_Plots/`` [Dead Link]
    .. [#Gaisser1] T. Gaisser,
       "Cosmic ray spectrum and composition > 100 TeV," IceCube Internal Report
       `icecube/201102004-v2
       <https://internal-apps.icecube.wisc.edu/reports/details.php?type=report&id=icecube%2F201102004>`_
       (2011).
    .. [#Gaisser2] T. Gaisser,
       "Spectrum of cosmic-ray nucleons and the atmospheric muon charge ratio,"
       `Astropart. Phys. 35, 801 (2012)
       <https://doi.org/10.1016/j.astropartphys.2012.02.010>`_.
       `arXiv:1111.6675v2 <https://arxiv.org/abs/1111.6675v2>`_.
    .. [#Aartsen] M. G. Aartsen et al. "Measurement of the cosmic ray energy spectrum with IceTop-73,"
       `Phys. Rev. D 88, 042004 (2013) <https://doi.org/10.1103/PhysRevD.88.042004>`_.
       `arXiv:1307.3795v1 <https://arxiv.org/abs/1307.3795v1>`_.
    .. [#Honda] M. Honda, T. Kajita, K. Kasahara, and S. Midorikawa,
       "New calculation of the atmospheric neutrino flux in a three-dimensional scheme,"
       `Phys. Rev. D 70, 043008 (2004) <https://doi.org/10.1103/PhysRevD.70.043008>`_.
       `astro-ph/0404457v4 <https://arxiv.org/abs/astro-ph/0404457v4>`_.
    .. [#Enberg] R. Enberg, M. H. Reno, I. Sarcevic, "Prompt neutrino fluxes from atmospheric charm,"
       `Phys. Rev. D 78, 043005 (2008) <https://doi.org/10.1103/PhysRevD.78.043005>`_.
       `arXiv:0806.0418 <https://arxiv.org/abs/0806.0418>`_.
    .. [#Thunman] M. Thunman, G. Ingelman, P. Gondolo,
       "Charm Production and High Energy Atmospheric Muon and Neutrino Fluxes,"
       `Astropart. Phys. 5, 309 (1996) <https://doi.org/10.1016/0927-6505(96)00033-3>`_.
       `hep-ph/9505417 <https://arxiv.org/abs/hep-ph/9505417>`_.
    .. [#GaisserStanevTilav] T. K. Gaisser, T. Stanev, and S. Tilav,
       "Cosmic ray energy spectrum from measurements of air showers,"
       `Frontiers of Physics 8, 748 (2013) <https://doi.org/10.1007/s11467-013-0319-7>`_.
       `arXiv:1303.3565 <https://arxiv.org/abs/1303.3565v1>`_.
    .. [#GSFDembinski] H. Dembinski et al.,
       "Data-driven model of the cosmic-ray flux and mass composition from 10 GeV to $10^{11} $ GeV."
       `arXiv:1711.11432 <https://arxiv.org/abs/1711.11432>`_.
    """
