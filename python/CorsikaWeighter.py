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
    return dict(energy     = weight_table.cols.PrimaryEnergy[:],
                type       = weight_table.cols.PrimaryType[:].astype(np.int32),
                cos_zenith = np.full(len(weight_table),np.nan),
                weight     = weight_table.cols.Weight[:])

corsika_flux_map = dict(E='energy',ptype='type')

CorsikaWeighter = make_weighter("I3CorsikaInfo", "CorsikaWeightMap",
                                corsika_surface_func,corsika_info_func,
                                corsika_event_data,
                                corsika_flux_map)
