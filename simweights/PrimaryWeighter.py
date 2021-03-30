import numpy as np
from . import VolumeCorrCylinder, PowerLaw, GenerationSurface
from .WeighterBase import Weighter, Null
from .utils import has_table, get_table

class PrimaryWeighter(Weighter):
    def __init__(self,infile):
        
        info_obj = "I3PrimaryInjectorInfo"
        if not has_table(infile, info_obj):
            raise RuntimeError(
                "File `{}` is Missing S-Frames table `I3PrimaryInjectorInfo`, "
                "this is required for PrimaryInjector files".format(infile.filename))

        info_table = get_table(infile, info_obj)

        surface = Null()
        for row in info_table:
            d = { x:row[x] for x in info_table.dtype.names }
            surface += self.get_surface(**d)

        self.surface = surface
        self.data = [infile]

    def get_surface(self, primary_type, n_events, cylinder_height,
                    cylinder_radius, min_zenith, max_zenith, min_energy,
                    max_energy, power_law_index, **kwargs):
        surface = VolumeCorrCylinder(cylinder_height, cylinder_radius, np.cos(max_zenith), np.cos(min_zenith))
        assert(power_law_index<0)
        spectrum = PowerLaw(power_law_index, min_energy, max_energy)        
        s = GenerationSurface(primary_type, n_events, spectrum, surface)
        return s

    def get_surface_params(self):
        return dict(particle_type=self.get_column('I3CorsikaWeight','type'),
                    energy=self.get_column('I3CorsikaWeight','energy'),
                    cos_zen=np.cos(self.get_column('I3CorsikaWeight','zenith')))

    def get_flux_params(self):
        return dict(E=self.get_column('I3CorsikaWeight','energy'),
                    ptype=self.get_column('I3CorsikaWeight','type'))

    def get_event_weight(self):
        return self.get_column('I3CorsikaWeight','weight')
