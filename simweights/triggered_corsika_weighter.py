import numpy as np

from .generation_surface import GenerationSurface
from .powerlaw import PowerLaw
from .spatial import NaturalRateCylinder
from .utils import Null, get_table, has_table
from .weighter import Weighter


class TriggeredCorsikaWeighter(Weighter):
    """
    Weighter for triggered (dynamic-stack) CORSIKA simulation.

    This simulation is generated with I3PrimaryInjector icetray Module.
    The cosmic-ray primaries are then propagated in the atmosphere with I3CORSIKAServer.
    These files have an S-Frame so there is no need to guess the number of jobs.
    These showers can be biased toward events with higher energy leading edge muons the weight from this
    process is stored in the frame as ``I3CorsikaWeight.weight`` which is used as the event weight.
    """

    event_map = dict(
        pdgid=("I3CorsikaWeight", "type"),
        energy=("I3CorsikaWeight", "energy"),
        zenith=("I3CorsikaWeight", "zenith"),
        event_weight=("I3CorsikaWeight", "weight"),
    )

    def __init__(self, infile):
        info_obj = "I3PrimaryInjectorInfo"
        if not has_table(infile, info_obj):
            raise RuntimeError(
                "File `{}` is Missing S-Frames table `I3PrimaryInjectorInfo`, "
                "this is required for PrimaryInjector files".format(infile.filename)
            )
        surface = Null()
        for row in get_table(infile, info_obj):
            surface += self._get_surface(row)
        super().__init__(surface, [infile])

    @staticmethod
    def _get_surface(smap):
        assert smap["power_law_index"] <= 0
        spatial = NaturalRateCylinder(
            smap["cylinder_height"],
            smap["cylinder_radius"],
            np.cos(smap["max_zenith"]),
            np.cos(smap["min_zenith"]),
        )
        spectrum = PowerLaw(smap["power_law_index"], smap["min_energy"], smap["max_energy"])
        return GenerationSurface(smap["primary_type"], smap["n_events"], spectrum, spatial)
