import numpy as np

from .generation_surface import GenerationSurface
from .powerlaw import PowerLaw
from .spatial import CircleInjector
from .utils import get_table
from .weighter import Weighter


def genie_surface(table):
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
            * GenerationSurface(row["primary_type"], spectrum, spatial)
        )
    return sum(surfaces)


def GenieWeighter(infile):
    # pylint: disable=invalid-name
    """
    Weighter for GENIE simulation

    Does not use S-Frames and stores the surface information in an I3MapStringDouble so that the user
    does not know how many jobs contributed to the current sample, so it needs to know the number of
    files. Nugen calculates the event weight in a column called ``TotalWeight`` which takes into account
    the netutrino cross-section, detector density, and distance traveled through the generation volume.
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
