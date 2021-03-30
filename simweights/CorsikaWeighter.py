import numpy as np
from . import VolumeCorrCylinder, PowerLaw, GenerationSurface
from .utils import get_column, get_table, get_constant_column, Null
from .WeighterBase import Weighter

class CorsikaWeighter(Weighter):
    def __init__(self, infile, nfiles):
        assert(nfiles is not None)            
        surface=Null()
        for i in self.read_weight_map(infile):
            surface += nfiles * self.get_surface(**i)
        self.surface = surface
        self.data = [infile]

    def get_surface(self, primary_type, n_events, oversampling, cylinder_height,
                    cylinder_radius, min_zenith, max_zenith, min_energy,
                    max_energy, power_law_index, **kwargs):
        surface = VolumeCorrCylinder(cylinder_height, cylinder_radius, np.cos(max_zenith), np.cos(min_zenith))
        assert(power_law_index < 0)
        spectrum = PowerLaw(power_law_index, min_energy, max_energy)        
        return GenerationSurface(primary_type, n_events*oversampling, spectrum, surface)

    def read_weight_map(self,infile):

        table = get_table(infile, 'CorsikaWeightMap')
        vals = {}
        vals["ParticleType"] = sorted(np.unique(get_column(table,'ParticleType').astype(int)))
        for x in ("CylinderLength","CylinderRadius","ThetaMin","ThetaMax",
              "OverSampling","Weight"):
            vals[x]=get_constant_column(get_column(table, x))

        for i in range(len(vals["ParticleType"])):
            mask = vals["ParticleType"][i] == get_column(table,"ParticleType")
            for x in ("NEvents","EnergyPrimaryMax","EnergyPrimaryMin","PrimarySpectralIndex"):
                if x not in vals:
                    vals[x]=[]
                vals[x].append(get_constant_column(get_column(table,x)[mask]))            

        return [ dict(primary_type    = p,                        
                      n_events        = vals['NEvents'][i],
                      oversampling    = vals["OverSampling"],
                      cylinder_height = vals['CylinderLength'],
                      cylinder_radius = vals['CylinderRadius'],
                      min_zenith      = vals['ThetaMin'],
                      max_zenith      = vals['ThetaMax'],
                      min_energy      = vals['EnergyPrimaryMin'][i],
                      max_energy      = vals['EnergyPrimaryMax'][i],
                      power_law_index = round(vals['PrimarySpectralIndex'][i],6))
                 for i, p in enumerate(vals["ParticleType"])]  

    def get_surface_params(self):
        return dict(particle_type = self.get_column('PolyplopiaPrimary', 'type'),
                    energy = self.get_column('PolyplopiaPrimary', 'energy'),
                    cos_zen = np.cos(self.get_column('PolyplopiaPrimary', 'zenith')))

    def get_flux_params(self):
        return dict(ptype = self.get_column('PolyplopiaPrimary', 'type'),
                    E = self.get_column('PolyplopiaPrimary', 'energy'))

    def get_event_weight(self):
        return np.ones_like(self.get_column('PolyplopiaPrimary', 'energy'))
