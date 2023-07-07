# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause


from typing import Any

import numpy as np

from ._generation_surface import GenerationSurface, generation_surface
from ._powerlaw import PowerLaw
from ._spatial import CircleInjector, SpatialDist, UniformSolidAngleCylinder
from ._utils import constcol, get_column, get_table, has_column
from ._weighter import Weighter


def nugen_spatial(table: Any) -> SpatialDist:
    """Inspect the ``I3MCWeightDict`` table of a nugen file to generate the spatial distribution.

    It will either return a CircleInjector or UniformSolidAngleCylinder
    depending on how the dataset was generated.
    """
    max_cos = np.cos(constcol(table, "MinZenith"))
    min_cos = np.cos(constcol(table, "MaxZenith"))

    # Before V04-01-00, nugen injection primaries on the surface of a circle perpendicular to the momentum
    # vector of the primary, this can be determined by checking `InjectionSurfaceR`. It will
    # be > 0 for circle injection and -1 for surface injection. In new versions >V6-00-00 it is not even
    # present indicating surface mode
    if has_column(table, "InjectionSurfaceR"):
        injection_radius = constcol(table, "InjectionSurfaceR")
    else:
        injection_radius = -1

    if injection_radius > 0:
        return CircleInjector(injection_radius, min_cos, max_cos)

    # Surface mode was added in V04-01-00 but the cylinder size was hard coded, `CylinderHeight` and
    # `CylinderRadius` were added after later V06-00-00. If they are not in the table then use the
    # hardcoded values
    cylinder_height = constcol(table, "CylinderHeight") if has_column(table, "CylinderHeight") else 1900
    cylinder_radius = constcol(table, "CylinderRadius") if has_column(table, "CylinderRadius") else 950
    return UniformSolidAngleCylinder(cylinder_height, cylinder_radius, min_cos, max_cos)


def nugen_spectrum(table: Any) -> PowerLaw:
    """Inspect the ``I3MCWeightDict`` table of a nugen file to generate to represent the energy spectrum."""
    min_energy = 10 ** constcol(table, "MinEnergyLog")
    max_energy = 10 ** constcol(table, "MaxEnergyLog")
    # the energy spectrum is always powerlaw however nugen uses positive value of `PowerLawIndex`
    # for negative slopes ie +2 means E**-2 injection spectrum
    power_law_index = -constcol(table, "PowerLawIndex")
    assert power_law_index <= 0
    return PowerLaw(power_law_index, min_energy, max_energy)


def nugen_surface(table: Any) -> GenerationSurface:
    """Inspect the `I3MCWeightDict` table object of a nugen file to generate a surface object."""
    spatial = nugen_spatial(table)
    spectrum = nugen_spectrum(table)
    pdgid = get_column(table, "PrimaryNeutrinoType")
    unique_pdgids = np.unique(pdgid)

    surfaces = []
    for pid in unique_pdgids:
        mask = pid == pdgid

        # neutrino-generator is usually produced with approximately equal proportions of nu and nubar
        # newer version will explicitly put the ratio in `TypeWeight` but for older version we
        # assume it is 0.5
        type_weight = constcol(table, "TypeWeight", mask) if has_column(table, "TypeWeight") else 0.5
        primary_type = constcol(table, "PrimaryNeutrinoType", mask)
        n_events = type_weight * constcol(table, "NEvents", mask)
        surfaces.append(n_events * generation_surface(int(primary_type), spectrum, spatial))
    ret = sum(surfaces)
    assert isinstance(ret, GenerationSurface)
    return ret


def NuGenWeighter(file_obj: Any, nfiles: float) -> Weighter:  # noqa: N802
    # pylint: disable=invalid-name
    """Weighter for neutrino-generator (NuGen) simulation.

    Does not use S-Frames and stores the surface information in an I3MapStringDouble so that the user
    does not know how many jobs contributed to the current sample, so it needs to know the number of
    files. NuGen calculates the event weight in a column called ``TotalWeight`` which takes into account
    the neutrino cross-section, detector density, and distance traveled through the generation volume.
    """
    weight_table = get_table(file_obj, "I3MCWeightDict")
    surface = nfiles * nugen_surface(weight_table)

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
    weighter.add_weight_column("event_weight", weighter.get_column("I3MCWeightDict", weightname))

    return weighter
