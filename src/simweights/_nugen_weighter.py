# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ._generation_surface import CompositeSurface, GenerationSurface
from ._powerlaw import PowerLaw
from ._spatial import CircleInjector, SpatialDist, UniformSolidAngleCylinder
from ._utils import constcol, get_column, get_table, has_column
from ._weighter import Weighter


def nugen_spatial(table: Any, mask: Any = None) -> SpatialDist:
    """Inspect the ``I3MCWeightDict`` table of a nugen file to generate the spatial distribution.

    It will either return a CircleInjector or UniformSolidAngleCylinder
    depending on how the dataset was generated.
    """
    max_cos = np.cos(constcol(table, "MinZenith", mask))
    min_cos = np.cos(constcol(table, "MaxZenith", mask))

    # Before V04-01-00, nugen injection primaries on the surface of a circle perpendicular to the momentum
    # vector of the primary, this can be determined by checking `InjectionSurfaceR`. It will
    # be > 0 for circle injection and -1 for surface injection. In new versions >V6-00-00 it is not even
    # present indicating surface mode
    injection_radius = constcol(table, "InjectionSurfaceR", mask) if has_column(table, "InjectionSurfaceR") else -1

    if injection_radius > 0:
        return CircleInjector(injection_radius, min_cos, max_cos)

    # Surface mode was added in V04-01-00 but the cylinder size was hard coded, `CylinderHeight` and
    # `CylinderRadius` were added after later V06-00-00. If they are not in the table then use the
    # hardcoded values
    cylinder_height = constcol(table, "CylinderHeight", mask) if has_column(table, "CylinderHeight") else 1900
    cylinder_radius = constcol(table, "CylinderRadius", mask) if has_column(table, "CylinderRadius") else 950
    return UniformSolidAngleCylinder(cylinder_height, cylinder_radius, min_cos, max_cos)


def nugen_spectrum(table: Any, mask: Any = None) -> PowerLaw:
    """Inspect the ``I3MCWeightDict`` table of a nugen file to generate to represent the energy spectrum."""
    min_energy = 10 ** constcol(table, "MinEnergyLog", mask)
    max_energy = 10 ** constcol(table, "MaxEnergyLog", mask)
    # the energy spectrum is always powerlaw however nugen uses positive value of `PowerLawIndex`
    # for negative slopes ie +2 means E**-2 injection spectrum
    power_law_index = -constcol(table, "PowerLawIndex", mask)
    assert power_law_index <= 0
    return PowerLaw(power_law_index, min_energy, max_energy)


class NuGenSurface(GenerationSurface):
    """Represents a surface on which GENIE simulation was generated on."""

    def get_epdf(self: "NuGenSurface", weight_cols: Mapping[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Get the extended pdf of a sample of GENIE."""
        return (
            self.nevents
            / weight_cols["TotalWeight"]
            * self.power_law.pdf(weight_cols["energy"])
            * self.spatial.pdf(weight_cols["cos_zen"])
        )


def nugen_surface(table: Any) -> CompositeSurface:
    """Inspect the `I3MCWeightDict` table object of a nugen file to generate a surface object."""
    pdgid = get_column(table, "PrimaryNeutrinoType")
    unique_pdgids = np.unique(pdgid)
    if len(unique_pdgids) == 0:
        msg = "`I3MCWeightDict` is empty. SimWeights cannot process this file"
        raise RuntimeError(msg)

    spatial = nugen_spatial(table)
    spectrum = nugen_spectrum(table)

    comp = CompositeSurface()
    for pid in unique_pdgids:
        mask = pid == pdgid

        # neutrino-generator is usually produced with approximately equal proportions of nu and nubar
        # newer version will explicitly put the ratio in `TypeWeight` but for older version we
        # assume it is 0.5
        type_weight = constcol(table, "TypeWeight", mask) if has_column(table, "TypeWeight") else 0.5
        primary_type = int(constcol(table, "PrimaryNeutrinoType", mask))
        n_events = type_weight * constcol(table, "NEvents", mask)
        comp.insert(NuGenSurface(primary_type, n_events, spectrum, spatial))
    return comp


def NuGenWeighter(file_obj: Any, nfiles: float) -> Weighter:  # noqa: N802
    # pylint: disable=invalid-name
    """Weighter for neutrino-generator (NuGen) simulation.

    Does not use S-Frames and stores the surface information in an I3MapStringDouble so that the user
    does not know how many jobs contributed to the current sample, so it needs to know the number of
    files. NuGen calculates the event weight in a column called ``TotalWeight`` which takes into account
    the neutrino cross-section, detector density, and distance traveled through the generation volume.
    """
    weight_table = get_table(file_obj, "I3MCWeightDict")
    surface = nugen_surface(weight_table)
    surface.scale(nfiles)

    weighter = Weighter([file_obj], surface)
    weighter.add_weight_column("energy", weighter.get_column("I3MCWeightDict", "PrimaryNeutrinoEnergy"))
    weighter.add_weight_column(
        "pdgid",
        weighter.get_column("I3MCWeightDict", "PrimaryNeutrinoType").astype(np.int32),
    )
    weighter.add_weight_column(
        "cos_zen",
        np.cos(weighter.get_column("I3MCWeightDict", "PrimaryNeutrinoZenith")),
    )

    # the event weight is stored in `TotalWeight` in newer simulation and
    # `TotalInteractionProbabilityWeight` in older simulation, so we are gonna need to let the weighter
    # know that
    if has_column(weight_table, "TotalWeight"):
        weightname = "TotalWeight"
    else:
        assert has_column(weight_table, "TotalInteractionProbabilityWeight")
        weightname = "TotalInteractionProbabilityWeight"
    weighter.add_weight_column("TotalWeight", weighter.get_column("I3MCWeightDict", weightname))

    return weighter
