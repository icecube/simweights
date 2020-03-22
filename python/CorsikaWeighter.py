import numpy as np
from . import UprightCylinder,ZenithBias,PowerLaw,GenerationSurface
from .utils import get_constant_column
from .WeighterBase import make_weighter

def corsika_surface_func(primary_type,n_events,oversampling,
            cylinder_height,cylinder_radius,min_zenith,max_zenith,
            min_energy,max_energy,power_law_index,
            **kwargs):
    surface = UprightCylinder(cylinder_height,cylinder_radius,
                              np.cos(max_zenith),np.cos(min_zenith),
                              ZenithBias.VolumeCorr)
    assert(power_law_index<0)
    spectrum = PowerLaw(power_law_index,min_energy,max_energy)        
    s= GenerationSurface( primary_type,n_events*oversampling,spectrum,surface)
    return s

def read_corsika_columns(table):
    vals = {}
    vals["ParticleType"] = sorted(np.unique(table.cols.ParticleType[:].astype(int)))
    for x in ("CylinderLength","CylinderRadius","ThetaMin","ThetaMax",
              "OverSampling","Weight"):
        vals[x]=get_constant_column(getattr(table.cols,x)[:])

    for i in range(len(vals["ParticleType"])):
        mask = vals["ParticleType"][i]==table.cols.ParticleType[:]
        for x in ("NEvents","EnergyPrimaryMax","EnergyPrimaryMin","PrimarySpectralIndex"):
            if x not in vals:
                vals[x]=[]
            vals[x].append(get_constant_column(getattr(table.cols,x)[:][mask]))            
    return vals

def corsika_info_func(table):
    v =read_corsika_columns(table)
    return [ dict(primary_type    = p,                        
                  n_events        = v['NEvents'][i],
                  oversampling    = v["OverSampling"],
                  cylinder_height = v['CylinderLength'],
                  cylinder_radius = v['CylinderRadius'],
                  min_zenith      = v['ThetaMin'],
                  max_zenith      = v['ThetaMax'],
                  min_energy      = v['EnergyPrimaryMin'][i],
                  max_energy      = v['EnergyPrimaryMax'][i],
                  power_law_index = round(v['PrimarySpectralIndex'][i],6))
             for i,p in enumerate(v["ParticleType"])]

def corsika_event_data(weight_table):
    return dict(PrimaryEnergy=weight_table.cols.PrimaryEnergy[:],
                PrimaryType  =weight_table.cols.PrimaryType[:].astype(np.int32),
                cos_zen      = np.full(len(weight_table),np.nan),
                Weight       = weight_table.cols.Weight[:])

corsika_flux_map = dict(E='PrimaryEnergy',ptype='PrimaryType')
corsika_surface_map = dict(energy='PrimaryEnergy',particle_type='PrimaryType',cos_zen='cos_zen')

CorsikaWeighter = make_weighter("I3CorsikaInfo", "CorsikaWeightMap",
                                corsika_surface_func,corsika_info_func,
                                corsika_event_data,
                                corsika_flux_map,corsika_surface_map,'Weight')



def primary_injector_surface_func(primary_type,n_events,
                                  cylinder_height,cylinder_radius,min_zenith,max_zenith,
                                  min_energy,max_energy,power_law_index,zenith_bias,
                                  **kwargs):
    surface = UprightCylinder(cylinder_height,cylinder_radius,
                              np.cos(max_zenith),np.cos(min_zenith),
                              ZenithBias(zenith_bias))

    assert(power_law_index<0)
    spectrum = PowerLaw(power_law_index,min_energy,max_energy)        
    s= GenerationSurface( primary_type,n_events,spectrum,surface)
    return s

def primary_injector_info_func(table):
    raise Error("primary injector must have 'S' Frame")

def primary_injector_event_data(weight_table):
    return dict(energy       = weight_table.cols.energy[:],
                type         = weight_table.cols.type[:].astype(np.int32),
                cos_zen      = np.cos(weight_table.cols.zenith[:]),
                weight       = np.full(len(weight_table),1.),
                )

primary_injector_flux_map = dict(E='energy',ptype='type')
primary_injector_surface_map = dict(energy='energy',particle_type='type',cos_zen='cos_zen')


PrimaryWeighter = make_weighter("I3PrimaryInjectorInfo", "I3MCPrimary",
                                primary_injector_surface_func,
                                primary_injector_info_func,
                                primary_injector_event_data,
                                primary_injector_flux_map,
                                primary_injector_surface_map,
                                'weight')
