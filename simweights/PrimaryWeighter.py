import numpy as np

from .utils import Null, get_table, has_table
from .WeighterBase import Weighter


class PrimaryWeighter(Weighter):
    def __init__(self, infile):
        info_obj = "I3PrimaryInjectorInfo"
        if not has_table(infile, info_obj):
            raise RuntimeError(
                "File `{}` is Missing S-Frames table `I3PrimaryInjectorInfo`, "
                "this is required for PrimaryInjector files".format(infile.filename)
            )
        surface = Null()
        for row in get_table(infile, info_obj):
            surface += self.get_surface(row)
        super().__init__(surface, [infile])

    def get_surface_params(self):
        return dict(
            particle_type=self.get_column("I3CorsikaWeight", "type"),
            energy=self.get_column("I3CorsikaWeight", "energy"),
            cos_zen=np.cos(self.get_column("I3CorsikaWeight", "zenith")),
        )

    def get_flux_params(self):
        return dict(
            E=self.get_column("I3CorsikaWeight", "energy"), ptype=self.get_column("I3CorsikaWeight", "type")
        )

    def get_event_weight(self):
        return self.get_column("I3CorsikaWeight", "weight")
