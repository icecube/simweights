import numpy as np
from . import GenerationSurface, VolumeCorrCylinder, PowerLaw
from .WeighterBase import make_weighter

def primary_injector_surface_func(
        primary_type,n_events, cylinder_height,cylinder_radius,min_zenith,
        max_zenith,min_energy,max_energy,power_law_index,**kwargs):
    
    surface = VolumeCorrCylinder(cylinder_height, cylinder_radius, np.cos(max_zenith), np.cos(min_zenith))
    assert(power_law_index<0)
    spectrum = PowerLaw(power_law_index, min_energy, max_energy)        
    s = GenerationSurface(primary_type, n_events, spectrum, surface)
    return s

def primary_injector_info_func(table):
    raise Exception("File `{}` is Missing S-Frames table `I3PrimaryInjectorInfo`, this is required for PrimaryInjector files"
                    .format(table.filename)
                    )

def primary_injector_event_data(weight_table):
    return dict(energy       = weight_table.cols.energy[:],
                type         = weight_table.cols.type[:].astype(np.int32),
                cos_zenith   = np.cos(weight_table.cols.zenith[:]),
                weight       = weight_table.cols.weight[:],
                )

primary_injector_flux_map = dict(E='energy', ptype='type')

PrimaryWeighter = make_weighter("I3PrimaryInjectorInfo",
                                "I3CorsikaWeight",
                                primary_injector_surface_func,
                                primary_injector_info_func,
                                primary_injector_event_data,
                                primary_injector_flux_map)
