import tables
import numpy as np
from pprint import pprint

from icecube.mcweights import NuGenWeighter

import nuflux
flux = nuflux.makeFlux('honda2006').getFlux
t=tables.open_file('asdf.hdf5','r')
weighter = NuGenWeighter(t)
pprint(weighter.surface.spectra)
weight=weighter.get_weights(flux)

nfiles = len(t.root.I3NuGenInfo)
energy = t.root.I3MCWeightDict.cols.PrimaryNeutrinoEnergy[:]
ptype = t.root.I3MCWeightDict.cols.PrimaryNeutrinoType[:].astype(np.int32)
cos_theta = np.cos(t.root.I3MCWeightDict.cols.PrimaryNeutrinoZenith[:])
oneweight = ( flux(ptype, energy, cos_theta)
              * t.root.I3MCWeightDict.cols.OneWeight[:]
              / (nfiles)
              / (t.root.I3MCWeightDict.cols.NEvents[:]/2))

ratio = weight/oneweight
print (ratio[:40])
print(all(abs(ratio-1.0) < 1e-14 ))
weighter2 = NuGenWeighter(t,sframe=False,nfiles=10)

print('EQ',weighter2.surface==weighter.surface)

r2 = weighter2.get_weights(flux)/oneweight
print (r2[:40])
print(all(abs(ratio-1.0)<1e-14))

pprint(weighter.surface.spectra)
#print('NFILES',weighter.nfiles)
pprint(weighter2.surface.spectra)

t.close()

exit()

from icecube.weighting import weighting
#generator = weighting.from_simprod(20789)
#generator *= nfiles
from icecube.weighting.fluxes import GaisserH3a
flux = GaisserH3a() 

fivecomp_args = {
    'LowerCutoffType': 'EnergyPerParticle',
    'UpperCutoffType': 'EnergyPerParticle',
    'ZenithBias': 'VOLUMECORR',
    'ZenithRange': [0.0, 1.5707963267948966],
    'emax': 1000000.0,
    'emin': 30000.0,
    'gamma': [-2.65, -2.6, -2.6, -2.6, -2.6],
    'height': 1600.0,
    'nevents': 1500000,
    'normalization': [5.0, 2.25, 1.1, 1.2, 1.0],
    'radius': 800.0}

#pprint(fivecomp_args)


#generator = weighting.FiveComponent(**fivecomp_args)

#pprint (generator.spectra)

hdf= tables.open_file("Level2_IC86.2016_corsika.020789.000000.hdf5",'r')

#print(dir( hdf.root.CorsikaWeightMap.cols))
energy = hdf.root.CorsikaWeightMap.cols.PrimaryEnergy[:]
ptype = hdf.root.CorsikaWeightMap.cols.PrimaryType[:]
#cos_zenith = np.cos(hdf.root.I3MCPrimary.cols.zenith[:1])
#g=generator(energy, ptype)
#weights = flux(energy, ptype)/generator(energy, ptype) 
gg,ww = np.load('weights_20789.np.npy')


from icecube.mcweights import PowerLaw,UprightCylinder,UprightCylinder,GenerationSurface,GenerationSurfaceCollection,ZenithBias,TeV

from icecube.mcweights import PDGCode as p

from icecube.mcweights.utils import get_constant_column
from icecube.mcweights.NuGenWeight import WeighterBase
#def get_equal_column(col):
#    val = col[0]
#    assert np.ndim(val)==0    
#    assert ((val==col).all())
#    return val



def read_corsika_columns(table):
    vals = {}
    vals["ParticleType"] = sorted(np.unique(table.cols.ParticleType[:].astype(int)))
    for x in ("CylinderLength","CylinderRadius","ThetaMin","ThetaMax",
              "OverSampling","AreaSum","FluxSum","Polygonato","Weight"):
        vals[x]=get_constant_column(getattr(table.cols,x)[:])

    for i in range(len(vals["ParticleType"])):
        mask = vals["ParticleType"][i]==table.cols.ParticleType[:]
        for x in ("NEvents","EnergyPrimaryMax","EnergyPrimaryMin","PrimarySpectralIndex"):
            if x not in vals:
                vals[x]=[]
            vals[x].append(get_constant_column(getattr(table.cols,x)[:][mask]))
    return vals


class CorsikaWeight(WeighterBase):
    particle_types = (p.PPlus,p.He4Nucleus,p.N14Nucleus,p.Al27Nucleus,p.Fe56Nucleus)
    _info_obj    = 'I3CorsikaInfo'
    _weight_obj  = 'CorsikaWeightMap'
    _unit        = 1.

    @staticmethod
    def info_from_weight_obj(table):
        v =read_corsika_columns(table)
        #pprint(v)

        zbias = ZenithBias.VolumeCorr #for all simprod corsika?
        surface = UprightCylinder(
            v['CylinderLength'],v['CylinderRadius'],
            np.cos(v['ThetaMax']), np.cos(v['ThetaMin']),zbias)
        assert (v['AreaSum']==surface.area(None))
        
        probs = []
        fs=[]
        
        for i,p in enumerate(v["ParticleType"]):

            assert (v['PrimarySpectralIndex'][i]<0)
            spectrum = PowerLaw(v['PrimarySpectralIndex'][i],
                                v['EnergyPrimaryMin'][i],
                                v['EnergyPrimaryMax'][i])

            s1= PowerLaw(v['PrimarySpectralIndex'][i],
                         v['EnergyPrimaryMin'][i]*TeV,
                         v['EnergyPrimaryMax'][i]*TeV)
            fs.append(s1.total_integral())
            
            
            probs.append(GenerationSurface(
                p,spectrum,surface,v['NEvents'][i]*v["OverSampling"]))

        print(sum(fs),v["FluxSum"])
        s = GenerationSurfaceCollection(probs)
        return s

    def get_gen(self,primary_type,n_events,oversampling,
                cylinder_height,cylinder_radius,min_zenith,max_zenith,
                min_energy,max_energy,power_law_index,
                **kwargs):
        pprint(kwargs)
        surface = UprightCylinder(cylinder_height,cylinder_radius,
                                  np.cos(max_zenith),np.cos(min_zenith),
                                  ZenithBias.VolumeCorr)
        assert(power_law_index<0)
        spectrum = PowerLaw(power_law_index,min_energy,max_energy)

        
        s= GenerationSurface( primary_type,n_events*oversampling,spectrum,surface)
        #pprint(s)
        return s


    def get_flux_params(self):
        c={}
        c['E']        = self.weight_table.cols.PrimaryEnergy[:]
        c['ptype'] = self.weight_table.cols.PrimaryType[:].astype(np.int32)        
        #c['cos_zen']       = None
        return c

    def get_surface_params(self):
        c={}
        c['energy']        = self.weight_table.cols.PrimaryEnergy[:]
        c['particle_type'] = self.weight_table.cols.PrimaryType[:].astype(np.int32)
        c['cos_zen']       = np.full(len(self.weight_table),6)
        #c['weight']        = self.weight_table.cols.TotalWeight[:]
        return c

    def get_interaction_prob(self):
        return self.weight_table.cols.Weight[:]

    def get_cols(self,infile):
        table = getattr(infile.root,self._weight_obj)
        print("####")
        print (table.colnames)
        c={}
        c['energy']    = table.cols.PrimaryEnergy[:]
        c['particle_type']     = table.cols.PrimaryType[:].astype(np.int32)
        #c['cos_zen'] = np.cos(table.cols.PrimaryNeutrinoZenith[:])
        c['weight']     = table.cols.Weight[:]
        pprint(c)
        return c
    
f = tables.open_file('corsika.py3.000000.N0000010.hdf5','r')
pprint(f.root.I3CorsikaInfo[:])
gen = CorsikaWeight(f)
pprint(gen.surface.spectra)

print( gen.get_weights(flux)[:1000])




#print(np.unique(f.root.CorsikaWeightMap.cols.Run,return_counts=True))
    
    
class FiveCompCorsika:
    particle_types = (p.PPlus,p.He4Nucleus,p.N14Nucleus,p.Al27Nucleus,p.Fe56Nucleus)

    info_obj    = 'I3NuGenInfo'
    weight_obj  = 'I3MCWeightDict'
    
    @staticmethod
    def from_weight_obj(table):
        v =read_corsika_columns(table)
        pprint(v)
        particle_types = v['ParticleType']
        if particle_types != [2212, 1000020040, 1000070140, 1000130270, 1000260560]:
            raise Exception("This file contains particle types other than the "
                            "standard 5 components: {}".format(particle_types))
        
        u={}
        u['cylinder_height']=v['CylinderLength']
        u['cylinder_radius']=v['CylinderRadius']
        u['min_zenith']=v['ThetaMin']
        u['max_zenith']=v['ThetaMax']        
        u['n_events']=int(sum(v['NEvents'])*v['OverSampling'])
        u['power_law_index']=[round(g,6) for g in v['PrimarySpectralIndex']]        
        u['zenith_bias']='VOLUMECORR'

        u['normalization']=[]
        for i in range(5):
            p = PowerLaw(v['PrimarySpectralIndex'][i],v['EnergyPrimaryMin'][i]*TeV,v['EnergyPrimaryMax'][i]*TeV)
            u['normalization'].append(v['NEvents'][i]/p.total_integral())
        nmin=min(u['normalization'])
        u['normalization']=[n/nmin for n in u['normalization']]
        
        if (v['EnergyPrimaryMin'][0]==np.array(v['EnergyPrimaryMin'])).all():
            u['lower_cutoff']='EnergyPerParticle'
            u['min_energy']=v['EnergyPrimaryMin'][0]
        else:
            raise Exception()

        if (v['EnergyPrimaryMax'][0]==np.array(v['EnergyPrimaryMax'])).all():
            u['upper_cutoff']='EnergyPerParticle'
            u['max_energy']=v['EnergyPrimaryMax'][0]
        else:
            raise Exception()
        return u
    

    @classmethod
    def get_gen(cls,n_events,normalization,power_law_index,min_energy,
                max_energy,cylinder_height,cylinder_radius,min_zenith,
                max_zenith,zenith_bias,lower_cutoff, upper_cutoff):

        if zenith_bias=='VOLUMECORR':            
            zbias = ZenithBias.VolumeCorr
        else:
            raise Exception        
        area = UprightCylinder(cylinder_height,cylinder_radius,
                               np.cos(max_zenith), np.cos(min_zenith),
                               zbias)

        if lower_cutoff=='EnergyPerParticle':
            lower_energy_scale = [1.]*5
        else:
            raise Exception()

        if upper_cutoff=='EnergyPerParticle':
            upper_energy_scale = [1.]*5
        else:
            raise Exception()
                
        fluxsums1 = []
        for i in range(5):
            mlo=lower_energy_scale[i]
            mhi =upper_energy_scale[i]
            g    = power_law_index[i]
            n    = normalization[i]    

            spectrum = PowerLaw(g,min_energy*mlo*TeV, max_energy*mhi*TeV)
            y = n* spectrum.total_integral()
            fluxsums1.append(y)


        nshowers1 = []
        collection = []
        for i in range(5):
            mlo=lower_energy_scale[i]
            mhi =upper_energy_scale[i]
            g    = power_law_index[i]
            n    = normalization[i]    
            ns = n_events*fluxsums1[i]/sum(fluxsums1)
            nshowers1.append(ns)
            spectrum = PowerLaw(g,min_energy*mlo, max_energy*mhi)
    
            gensurf = GenerationSurface(cls.particle_types[i],ns,spectrum,area)
            collection.append(gensurf)

        return GenerationSurfaceCollection(collection)


a1 = FiveCompCorsika.from_weight_obj((hdf.root.CorsikaWeightMap))

pprint(a1)
pprint(FiveCompCorsika.get_gen(**a1).spectra)

args={'cylinder_height': 1600.0,
      'cylinder_radius': 800.0,
      'max_energy': 1000000.0,
      'max_zenith': 1.5707963267948966,
      'min_energy': 30000.0,
      'min_zenith': 0.0,
      'n_events': 1500000,
      'normalization': [5.0, 2.25, 1.1, 1.2, 1.0],
      'power_law_index': [-2.65, -2.6, -2.6, -2.6, -2.6],
      'lower_cutoff': 'EnergyPerParticle',      
      'upper_cutoff': 'EnergyPerParticle',
      'zenith_bias': 'VOLUMECORR'}

pprint(args)

gen1 = FiveCompCorsika.get_gen(**args)
pprint(gen1.spectra)

g1=gen1(ptype,energy,np.full(len(ptype),np.nan))
weights1 = flux(energy, ptype)/g1


print(weights1.sum())

#print(g1[:10])
#print(weights1[:10])

print(weights1[:1000])

print((gg-g1< 1e-16).all())
print((weights1-ww <1e-16).all() )

#hdf.close()
