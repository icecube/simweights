from typing import Any, Iterable, Mapping

import numpy as np

from .generation_surface import GenerationSurface, NullSurface, generation_surface
from .powerlaw import PowerLaw
from .spatial import CircleInjector
from .utils import get_table
from .weighter import Weighter


def genie_surface(table: Iterable[Mapping[str, float]]) -> GenerationSurface:
    """
    Inspect the rows of a GENIE S-Frame table object to generate a surface object.
    """
    surfaces = []
    for row in table:
        assert row["power_law_index"] >= 0
        spatial = CircleInjector(
            row["cylinder_radius"],
            np.cos(row["max_zenith"]),
            np.cos(row["min_zenith"]),
        )
        spectrum = PowerLaw(-row["power_law_index"], row["min_energy"], row["max_energy"])
        surfaces.append(
            row["n_flux_events"]
            / row["global_probability_scale"]
            * generation_surface(int(row["primary_type"]), spectrum, spatial)
        )
    return sum(surfaces, NullSurface)


def GenieWeighter(infile: Any) -> Weighter:
    # pylint: disable=invalid-name
    """
    Weighter for GENIE simulation

    Reads ``I3GenieInfo`` from S-Frames and ``I3GenieResult`` from Q-Frames.
    """

    weight_table = get_table(infile, "I3GenieInfo")
    surface = genie_surface(weight_table)

    event_map = dict(
        energy=("I3GenieResult", "Ev"),
        pdgid=("I3GenieResult", "neu"),
        zenith=None,
        event_weight=("I3GenieResult", "wght"),
    )

    return Weighter([(infile, event_map)], surface)
