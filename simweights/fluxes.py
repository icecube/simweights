"""
A collection of cosmic ray flux paramerizations.

This files contains the same Cosmic Ray flux models as :file:`weighting/python/fluxes.py`.
However they have been refactored to:

* Use PDG particle codes natively instad of CORSIKA code
* Use :py:func:`numpy.piecewise` instad of :py:mod:`numexpr`
* Follow :py:mod:`numpy` broadcasting rules

"""
from enum import IntEnum

from numpy import asarray, broadcast_arrays, exp, piecewise, sqrt


class PDGCode(IntEnum):
    """
    Enumeration of the PDG particle numbering scheme by the Particle Data Group (PDG) for cosmic-ray
    primaries

    The PDG assigns a unique code to each type of particle. The numbering includes all known elementary
    particles, composite particles, and atomic nuclei. However this enumeration is only used for cosmic-ray
    flux models and is limited to particle types in these models.
    """

    # pylint: disable=invalid-name
    PPlus = 2212
    He4Nucleus = 1000020040
    Li7Nucleus = 1000030070
    Be9Nucleus = 1000040090
    B11Nucleus = 1000050110
    C12Nucleus = 1000060120
    N14Nucleus = 1000070140
    O16Nucleus = 1000080160
    F19Nucleus = 1000090190
    Ne20Nucleus = 1000100200
    Na23Nucleus = 1000110230
    Mg24Nucleus = 1000120240
    Al27Nucleus = 1000130270
    Si28Nucleus = 1000140280
    P31Nucleus = 1000150310
    S32Nucleus = 1000160320
    Cl35Nucleus = 1000170350
    Ar40Nucleus = 1000180400
    K39Nucleus = 1000190390
    Ca40Nucleus = 1000200400
    Sc45Nucleus = 1000210450
    Ti48Nucleus = 1000220480
    V51Nucleus = 1000230510
    Cr52Nucleus = 1000240520
    Mn55Nucleus = 1000250550
    Fe56Nucleus = 1000260560


def corsika_to_pdg(cid):
    """
    Convert CORSIKA particle code to particle data group (PDG) Monte Carlo
    numbering scheme.

    Note:
        This function will only convert codes that correspond to
        nuclei needed for the flux models in this module. That includes PPlus (14)
        and He4Nucleus (402) through Fe56Nucleus (5626).

    Args:
        code (array_like): CORSIKA codes

    Returns:
        array_like: PDG codes
    """
    cid = asarray(cid)
    return piecewise(
        cid,
        [cid == 14, (cid >= 100) & (cid <= 9999)],
        [2212, lambda c: 1000000000 + 10000 * (c % 100) + 10 * (c // 100)],
    )


# pylint: disable=too-few-public-methods


class CosmicRayFlux:
    """
    Base class for cosmic ray fluxes that uses :py:func:`numpy.piecewise` for effient
    mathematical evaluation

    Derived must set `pdgids` to enumerate the particle types in this model.
    :py:func:`_funcs` must be set to a list of functions to be called for each particle
    type. :py:func:`_condition()` can be overidden if additional piecewise conditions are
    desired.
    """

    pdgids = []
    _funcs = []

    def _condition(self, energy, pdgid):
        # pylint: disable=unused-argument
        return [pdgid == p for p in self.pdgids]

    def __call__(self, energy, pdgid):
        energy, pdgid = broadcast_arrays(energy, pdgid)
        pcond = self._condition(energy, pdgid)
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
        lambda E: 11776.445965025136 * E ** -2.71 * (1 + (E / (4.49e6 * 1)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 4749.371132996256 * E ** -2.64 * (1 + (E / (4.49e6 * 2)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 86.7088317618298 * E ** -2.54 * (1 + (E / (4.49e6 * 3)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 84.29044403584496 * E ** -2.75 * (1 + (E / (4.49e6 * 4)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 633.6114770238042 * E ** -2.95 * (1 + (E / (4.49e6 * 5)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 1012.2921411827233 * E ** -2.66 * (1 + (E / (4.49e6 * 6)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 339.6783461252935 * E ** -2.72 * (1 + (E / (4.49e6 * 7)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 1721.4707679448027 * E ** -2.68 * (1 + (E / (4.49e6 * 8)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 38.536639802016566 * E ** -2.69 * (1 + (E / (4.49e6 * 9)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 382.61133470722905 * E ** -2.64 * (1 + (E / (4.49e6 * 10)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 72.00644098601636 * E ** -2.66 * (1 + (E / (4.49e6 * 11)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 666.2427806532402 * E ** -2.64 * (1 + (E / (4.49e6 * 12)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 109.82414739246525 * E ** -2.66 * (1 + (E / (4.49e6 * 13)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 1415.5104103909828 * E ** -2.75 * (1 + (E / (4.49e6 * 14)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 31.722233983367293 * E ** -2.69 * (1 + (E / (4.49e6 * 15)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 102.29054260257044 * E ** -2.55 * (1 + (E / (4.49e6 * 16)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 32.23645896660968 * E ** -2.68 * (1 + (E / (4.49e6 * 17)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 69.53545126418338 * E ** -2.64 * (1 + (E / (4.49e6 * 18)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 47.77105028396875 * E ** -2.65 * (1 + (E / (4.49e6 * 19)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 185.06203553374286 * E ** -2.70 * (1 + (E / (4.49e6 * 20)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 25.28561864152123 * E ** -2.64 * (1 + (E / (4.49e6 * 21)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 76.39737621929389 * E ** -2.61 * (1 + (E / (4.49e6 * 22)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 48.981193059270424 * E ** -2.63 * (1 + (E / (4.49e6 * 23)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 139.16784695018254 * E ** -2.67 * (1 + (E / (4.49e6 * 24)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 32.384244406763116 * E ** -2.46 * (1 + (E / (4.49e6 * 25)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 1201.2410569254007 * E ** -2.59 * (1 + (E / (4.49e6 * 26)) ** 1.9) ** (-2.1 / 1.9),
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
        lambda E: 11776.445965025136 * E ** -2.71 * (1 + (E / (4.49e6 * 1)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 4749.371132996256 * E ** -2.64 * (1 + (E / (4.49e6 * 2)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 3552.5893555039243 * E ** -2.68 * (1 + (E / (4.49e6 * 7)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 3233.6058556071825 * E ** -2.67 * (1 + (E / (4.49e6 * 13)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 1197.9991050096223 * E ** -2.58 * (1 + (E / (4.49e6 * 26)) ** 1.9) ** (-2.1 / 1.9),
    ]


class Hoerandel_IT(CosmicRayFlux):
    """
    Modified 5-component Hoerandel spectrum with N and Al replaced by O.
    """

    # pylint: disable=invalid-name
    pdgids = [PDGCode.PPlus, PDGCode.He4Nucleus, PDGCode.O16Nucleus, PDGCode.Fe56Nucleus]
    _funcs = [
        lambda E: 11776.445965025136 * E ** -2.71 * (1 + (E / (4.49e6 * 1)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 4749.371132996256 * E ** -2.64 * (1 + (E / (4.49e6 * 2)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 7017.460455316394 * E ** -2.68 * (1 + (E / (4.49e6 * 8)) ** 1.9) ** (-2.1 / 1.9),
        lambda E: 1197.9991050096223 * E ** -2.58 * (1 + (E / (4.49e6 * 26)) ** 1.9) ** (-2.1 / 1.9),
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
        lambda E: 7860 * E ** (-2.66) * exp(-E / (4e6 * 1)),
        lambda E: 3550 * E ** (-2.58) * exp(-E / (4e6 * 2)),
        lambda E: 2200 * E ** (-2.63) * exp(-E / (4e6 * 7)),
        lambda E: 1430 * E ** (-2.67) * exp(-E / (4e6 * 13)),
        lambda E: 2120 * E ** (-2.63) * exp(-E / (4e6 * 26)),
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
        lambda E: 7860 * E ** -2.66 * exp(-E / (4e6 * 1))
        + 20.0 * E ** -2.4 * exp(-E / (3e7 * 1))
        + 1.70 * E ** -2.4 * exp(-E / (2e9 * 1)),
        lambda E: 3550 * E ** -2.58 * exp(-E / (4e6 * 2))
        + 20.0 * E ** -2.4 * exp(-E / (3e7 * 2))
        + 1.70 * E ** -2.4 * exp(-E / (2e9 * 2)),
        lambda E: 2200 * E ** -2.63 * exp(-E / (4e6 * 7))
        + 13.4 * E ** -2.4 * exp(-E / (3e7 * 7))
        + 1.14 * E ** -2.4 * exp(-E / (2e9 * 7)),
        lambda E: 1430 * E ** -2.67 * exp(-E / (4e6 * 13))
        + 13.4 * E ** -2.4 * exp(-E / (3e7 * 13))
        + 1.14 * E ** -2.4 * exp(-E / (2e9 * 13)),
        lambda E: 2120 * E ** -2.63 * exp(-E / (4e6 * 26))
        + 13.4 * E ** -2.4 * exp(-E / (3e7 * 26))
        + 1.14 * E ** -2.4 * exp(-E / (2e9 * 26)),
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
        lambda E: 7860 * E ** -2.66 * exp(-E / (4e6 * 1))
        + 20.0 * E ** -2.4 * exp(-E / (3e7 * 1))
        + 200 * E ** -2.6 * exp(-E / 6e10),
        lambda E: 3550 * E ** -2.58 * exp(-E / (4e6 * 2)) + 20.0 * E ** -2.4 * exp(-E / (3e7 * 2)),
        lambda E: 2200 * E ** -2.63 * exp(-E / (4e6 * 7)) + 13.4 * E ** -2.4 * exp(-E / (3e7 * 7)),
        lambda E: 1430 * E ** -2.67 * exp(-E / (4e6 * 13)) + 13.4 * E ** -2.4 * exp(-E / (3e7 * 13)),
        lambda E: 2120 * E ** -2.63 * exp(-E / (4e6 * 26)) + 13.4 * E ** -2.4 * exp(-E / (3e7 * 26)),
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
        lambda E: 7860 * E ** -2.66 * exp(-E / (4e6 * 1))
        + 20.0 * E ** -2.4 * exp(-E / (3e7 * 1))
        + 200 * E ** -2.6 * exp(-E / 6e10),
        lambda E: 3550 * E ** -2.58 * exp(-E / (4e6 * 2)) + 20.0 * E ** -2.4 * exp(-E / (3e7 * 2)),
        lambda E: 2200 * E ** -2.63 * exp(-E / (4e6 * 7))
        + 13.4 * E ** -2.4 * exp(-E / (3e7 * 7))
        + 1430 * E ** -2.67 * exp(-E / (4e6 * 13))
        + 13.4 * E ** -2.4 * exp(-E / (3e7 * 13)),
        lambda E: 2120 * E ** -2.63 * exp(-E / (4e6 * 26)) + 13.4 * E ** -2.4 * exp(-E / (3e7 * 26)),
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
        lambda E: (14900) * (E + 2.15 * exp(-0.21 * sqrt(E))) ** (-2.74),
        lambda E: (14900) * (100 ** (2.71 - 2.74)) * (E + 2.15 * exp(-0.21 * sqrt(E))) ** (-2.71),
        lambda E: (600 / 4.02) * (E / 4.02 + 1.25 * exp(-0.14 * sqrt(E / 4.02))) ** (-2.64),
        lambda E: (33.2 / 14.07) * (E / 14.07 + 0.97 * exp(-0.01 * sqrt(E / 14.07))) ** (-2.60),
        lambda E: (34.2 / 27.13) * (E / 27.13 + 2.14 * exp(-0.01 * sqrt(E / 27.13))) ** (-2.79),
        lambda E: (4.45 / 56.26) * (E / 56.26 + 3.07 * exp(-0.41 * sqrt(E / 56.26))) ** (-2.68),
    ]

    def _condition(self, energy, pdgid):
        return [
            (pdgid == 2212) * (energy < 100),
            (pdgid == 2212) * (energy >= 100),
            pdgid == 1000020040,
            pdgid == 1000070140,
            pdgid == 1000130270,
            pdgid == 1000260560,
        ]


class TIG1996(CosmicRayFlux):
    r"""
    Spectrum used to calculate prompt neutrino fluxes in Enberg et al. (2008)\ [#Enberg]_ (Eq. 30).
    The parameterization was taken directly from an earlier paper by Thunman et al\ [#Thunman]_.
    Only the nucleon flux was given, so for simplicity we treat it as a proton-only flux.
    """
    pdgids = [PDGCode.PPlus]
    _funcs = [lambda E: 1.70e4 * E ** -2.7, lambda E: 1.74e6 * E ** -3.0, 0]

    def _condition(self, energy, pdgid):
        return [(pdgid == 2212) * (energy < 5e6), (pdgid == 2212) * (energy >= 5e6)]


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
        lambda E: 7000 * E ** -2.66 * exp(-E / 1.2e5)
        + 150 * E ** -2.4 * exp(-E / 4e6)
        + 14 * E ** -2.4 * exp(-E / 1.3e9),
        lambda E: 3200 * E ** -2.58 * exp(-E / 1.2e5 / 2) + 65 * E ** -2.3 * exp(-E / 4e6 / 2),
        lambda E: 100 * E ** -2.40 * exp(-E / 1.2e5 / 7) + 6 * E ** -2.3 * exp(-E / 4e6 / 7),
        lambda E: 130 * E ** -2.40 * exp(-E / 1.2e5 / 13) + 7 * E ** -2.3 * exp(-E / 4e6 / 13),
        lambda E: 60 * E ** -2.30 * exp(-E / 1.2e5 / 26)
        + 2.3 * E ** -2.2 * exp(-E / 4e6 / 26)
        + 0.025 * E ** -2.2 * exp(-E / 1.3e9 / 26),
    ]


class FixedFractionFlux(CosmicRayFlux):
    """
    Total energy per particle flux flux split among mass groups with a constant fraction.

    By default, this takes as basis the flux from Gaisser H4a summed over all its mass groups,
    then multiplies it by the given fraction. This is a quick way to consider different
    weightings for systematic checks.
    """

    def __init__(self, fractions, basis=GaisserH4a_IT(), normalized=True):
        """
        :param fractions: A dictionary of fractions. They must add up to one and they
        should correspond to the pdgids in basis

        :type fractions: a dictionary with dataclasses.ParticleType as keys
        """
        self.flux = basis
        fluxes = {int(k): 0 for k in basis.pdgids}
        fluxes.update({int(k): v for k, v in fractions.items()})
        self.pdgids = list(fluxes.keys())
        self.fracs = list(fluxes.values())
        if normalized:
            assert sum(self.fracs) == 1.0

    def __call__(self, energy, pdgid):
        """
        :param E: particle energy in GeV
        :param pdgid: particle type code
        :type pdgid: int
        """
        energy, pdgid = broadcast_arrays(energy, pdgid)
        fluxsum = sum(self.flux(energy, p) for p in self.pdgids)
        cond = self._condition(energy, pdgid)
        return fluxsum * piecewise(energy, cond, self.fracs)


class _references:
    """
    .. [#Hoerandel] J. R. Hörandel, "On the knee in the energy spectrum of cosmic rays,"
       `Astropart. Phys. 19, 193 (2003)
       <http://dx.doi.org/10.1016/S0927-6505(02)00198-6>`_.
       `astro-ph/0210453 <https://arxiv.org/abs/astro-ph/0210453>`_.
    .. [#Becherini] Y. Becherini et al.,
       "A parameterisation of single and multiple muons in the deep water or ice,"
       `Astropart. Phys. 25, 1 (2006)
       <http://dx.doi.org/10.1016/j.astropartphys.2005.10.005>`_.
       `hep-ph/0507228 <https://arxiv.org/abs/hep-ph/0507228>`_.
    .. [#Schoenwald] A. Schoenwald, http://www.ifh.de/~arnes/Forever/Hoerandel_Plots/ [Dead Link]
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
    """
