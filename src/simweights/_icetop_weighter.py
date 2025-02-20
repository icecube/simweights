# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause


from typing import Any

import numpy as np

from ._generation_surface import GenerationSurface, generation_surface
from ._powerlaw import PowerLaw
from ._spatial import NaturalRateCylinder
from ._utils import get_column, get_table
from ._weighter import Weighter


def sframe_icetop_surface(table: Any) -> GenerationSurface:
    """Inspect the rows of a I3TopInjectorInfo S-Frame table object to generate a surface object."""
    surfaces = []

    n_events = get_column(table, "n_events")
    power_law_index = get_column(table, "power_law_index")
    min_energy = get_column(table, "min_energy")
    max_energy = get_column(table, "max_energy")
    sampling_radius = get_column(table, "sampling_radius")
    max_zenith = np.cos(get_column(table, "max_zenith"))
    min_zenith = np.cos(get_column(table, "min_zenith"))
    primary_type = get_column(table, "primary_type")

    for i, nevents in enumerate(n_events):
        assert power_law_index[i] <= 0
        spectrum = PowerLaw(  # pylint: disable=duplicate-code
            power_law_index[i],
            min_energy[i],
            max_energy[i],
            "energy",
        )
        spatial = NaturalRateCylinder(
            0,  # set cylinder height to 0 to get simple surface plane
            sampling_radius[i],
            max_zenith[i],
            min_zenith[i],
            "cos_zen",
        )
        pdgid = int(primary_type[i])
        surfaces.append(nevents * generation_surface(pdgid, spectrum, spatial))
    retval = sum(surfaces)
    assert isinstance(retval, GenerationSurface)
    return retval


def IceTopWeighter(file_obj: Any) -> Weighter:  # noqa: N802
    # pylint: disable=invalid-name
    """Weighter for IceTop CORSIKA simulation made with I3TopSimulator.cxx."""
    surface = sframe_icetop_surface(get_table(file_obj, "I3TopInjectorInfo"))
    weighter = Weighter([file_obj], surface)

    pdgid = weighter.get_column("MCPrimary", "type")
    weighter.add_weight_column("pdgid", pdgid)
    weighter.add_weight_column("energy", weighter.get_column("MCPrimary", "energy"))
    weighter.add_weight_column("cos_zen", np.cos(weighter.get_column("MCPrimary", "zenith")))
    return weighter
