import numpy as np
from . import UprightCylinder,ZenithBias,PowerLaw,GenerationSurface,GenerationSurfaceCollection
from .utils import get_constant_column
from .WeighterBase import make_weighter
#from pprint import pprint
from .units import cm2

nugen_weight_name = 'weight'
nugen_map = dict(energy='energy',particle_type='particle_type',cos_zen='cos_zen')
def nugen_event_data(weight_table):
    return dict(energy        = weight_table.cols.PrimaryNeutrinoEnergy[:],
                particle_type = weight_table.cols.PrimaryNeutrinoType[:].astype(np.int32),
                cos_zen       = np.cos(weight_table.cols.PrimaryNeutrinoZenith[:]),
                weight        = weight_table.cols.TotalWeight[:]*cm2,
    )
def nugen_read_weight_obj(table):
    return [dict(
        n_events        = get_constant_column(table.cols.NEvents),
        primary_type    = get_constant_column(abs(table.cols.PrimaryNeutrinoType[:])),
        cylinder_height = get_constant_column(table.cols.CylinderHeight),
        cylinder_radius = get_constant_column(table.cols.CylinderRadius),
        flavor_fraction = 0.5,
        min_azimuth     = get_constant_column(table.cols.MinAzimuth),               
        max_azimuth     = get_constant_column(table.cols.MaxAzimuth),
        min_energy      = 10**get_constant_column(table.cols.MinEnergyLog),               
        max_energy      = 10**get_constant_column(table.cols.MaxEnergyLog),
        min_zenith      = get_constant_column(table.cols.MinZenith),               
        max_zenith      = get_constant_column(table.cols.MaxZenith),
        power_law_index = get_constant_column(table.cols.PowerLawIndex),               
    )]
def nugen_surface(cylinder_height,cylinder_radius,flavor_fraction,max_azimuth,
            max_energy, max_zenith, min_azimuth, min_energy, min_zenith,
            n_events, power_law_index, primary_type, **kwargs):
    surface = UprightCylinder(cylinder_height, cylinder_radius,
                              np.cos(max_zenith),np.cos(min_zenith),
                              ZenithBias.VolumeDet)
    assert(power_law_index>0)
    powerlaw = PowerLaw(-power_law_index,min_energy,max_energy)
    probs = [GenerationSurface( primary_type,n_events*(  flavor_fraction),powerlaw,surface),
             GenerationSurface(-primary_type,n_events*(1-flavor_fraction),powerlaw,surface)]
    return GenerationSurfaceCollection(probs)

NuGenWeighter = make_weighter("I3NuGenInfo","I3MCWeightDict",nugen_surface,
                              nugen_read_weight_obj,nugen_event_data,
                              nugen_map,nugen_map,'weight')
