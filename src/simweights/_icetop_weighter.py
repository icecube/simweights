# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause


from typing import Any

import numpy as np

from ._generation_surface import GenerationSurface, generation_surface
from ._powerlaw import PowerLaw
from ._spatial import CircleInjector
from ._utils import get_column, get_table
from ._weighter import Weighter


def sframe_icetop_surface(table: Any) -> GenerationSurface:
    """Inspect the rows of a I3TopInjectorInfo S-Frame table object to generate a surface object."""
    surfaces = []

    for i in range(len(get_column(table, "n_events"))):
        assert get_column(table, "power_law_index")[i] <= 0
        spectrum = PowerLaw(
            get_column(table, "power_law_index")[i],
            get_column(table, "min_energy")[i],
            get_column(table, "max_energy")[i],
        )
        spatial = CircleInjector(
            get_column(table, "sampling_radius")[i],
            np.cos(get_column(table, "max_zenith")[i]),
            np.cos(get_column(table, "min_zenith")[i]),
        )
        surfaces.append(
            get_column(table, "n_events")[i]
            * generation_surface(int(get_column(table, "primary_type")[i]), spectrum, spatial),
        )
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
    weighter.add_weight_column("event_weight", np.ones_like(pdgid))

    return weighter
