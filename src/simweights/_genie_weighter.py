# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause


from typing import Any, Iterable, Mapping

import numpy as np

from ._generation_surface import GenerationSurface, generation_surface
from ._powerlaw import PowerLaw
from ._spatial import CircleInjector
from ._utils import Column, Const, get_column, get_table
from ._weighter import Weighter


def genie_surface(table: Iterable[Mapping[str, float]]) -> GenerationSurface:
    """Inspect the rows of a GENIE S-Frame table object to generate a surface object."""
    surfaces = []

    for i in range(len(get_column(table, "n_flux_events"))):
        assert get_column(table, "power_law_index")[i] >= 0
        spatial = CircleInjector(
            get_column(table, "cylinder_radius")[i],
            np.cos(get_column(table, "max_zenith")[i]),
            np.cos(get_column(table, "min_zenith")[i]),
        )
        spectrum = PowerLaw(
            -get_column(table, "power_law_index")[i],
            get_column(table, "min_energy")[i],
            get_column(table, "max_energy")[i],
            "energy",
        )
        pdgid = int(get_column(table, "primary_type")[i])
        nevents = get_column(table, "n_flux_events")[i]
        global_probability_scale = get_column(table, "global_probability_scale")[i]
        surfaces.append(
            nevents
            * generation_surface(pdgid, Const(1 / spatial.etendue / global_probability_scale), Column("wght"), spectrum),
        )
    retval = sum(surfaces)
    assert isinstance(retval, GenerationSurface)
    return retval


def GenieWeighter(file_obj: Any) -> Weighter:  # noqa: N802
    # pylint: disable=invalid-name
    """Weighter for GENIE simulation.

    Reads ``I3GenieInfo`` from S-Frames and ``I3GenieResult`` from Q-Frames.
    """
    weight_table = get_table(file_obj, "I3GenieInfo")
    surface = genie_surface(weight_table)

    weighter = Weighter([file_obj], surface)
    weighter.add_weight_column("pdgid", weighter.get_column("I3GenieResult", "neu").astype(np.int32))
    weighter.add_weight_column("energy", weighter.get_column("I3GenieResult", "Ev"))
    weighter.add_weight_column("wght", weighter.get_column("I3GenieResult", "wght"))

    return weighter
