import numpy as np

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

    def _get_surface_params(self):
        return dict(
            particle_type=self.get_column("I3CorsikaWeight", "type"),
            energy=self.get_column("I3CorsikaWeight", "energy"),
            cos_zen=np.cos(self.get_column("I3CorsikaWeight", "zenith")),
        )

    def _get_flux_params(self):
        return dict(
            E=self.get_column("I3CorsikaWeight", "energy"), ptype=self.get_column("I3CorsikaWeight", "type")
        )

    def _get_event_weight(self):
        # this is the weight from the leading edge muon shower bias
        return self.get_column("I3CorsikaWeight", "weight")
