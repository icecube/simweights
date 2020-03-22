import pickle,tables
from pprint import pprint
import numpy as np
import pylab as plt

from icecube.weighting.fluxes import GaisserH3a
from icecube.weighting import weighting
from kjm.hist import hist
from icecube.mcweights import CorsikaWeighter
from icecube.mcweights.WeighterBase import NullWeighter

with open('simprod.pkl', 'rb') as f:
    gen = pickle.load(f)

for g in gen.values():
    pprint(g.spectra)
    
datasets = [
    (None ,tables.open_file('Level2_IC86.2018_data_Run00132638_Subrun00000000_00000000.hdf5'),'data'),
    (20881,tables.open_file('Level2_IC86.2016_corsika.020881.000000.hdf5'),'C low'),    
    (20789,tables.open_file('Level2_IC86.2016_corsika.020789.000000.hdf5'),'C mid'),
    (20848,tables.open_file('Level2_IC86.2016_corsika.020848.000000.hdf5'),'C high'),    
    ]

for r,d, l in datasets:
    if hasattr(d.root,'CorsikaWeightMap'):
        print(l,r,
              d.root.CorsikaWeightMap.cols.Run[0],
              ':'.join(str(s) for s in np.unique(d.root.CorsikaWeightMap.cols.EnergyPrimaryMin)),
              ':'.join(str(s) for s in np.unique(d.root.CorsikaWeightMap.cols.EnergyPrimaryMax)))
        
t1=datasets[0][1].root.I3EventHeader.cols.time_start_utc_daq[0]
t2=datasets[0][1].root.I3EventHeader.cols.time_start_utc_daq[-1]

livetime= (t2-t1)/1e10
print("LIVE", livetime)

plt.figure()

g = weighting.Null()
e = []
p = []
m = np.array([],dtype=bool)
q =[]
z=[]

qtot_r=[10,4e4]
flux = GaisserH3a()
www = NullWeighter()
print(www)

for r,d,l in datasets:
    mf= d.root.FilterMask.cols.MuonFilter_13[:][:,0].astype(bool)
    if r:
        energy = d.root.CorsikaWeightMap.cols.PrimaryEnergy[:]
        ptype = d.root.CorsikaWeightMap.cols.PrimaryType[:].astype(int)
        weights = flux(energy, ptype)/gen[r](energy, ptype)/(1600./1200)/(800./600)

        g+=gen[r]
        e=np.r_[e,energy]
        p=np.r_[p,ptype]
        m=np.r_[m,mf]
        q=np.r_[q,d.root.Homogenized_QTot.cols.value[:]]
        z=np.r_[z,d.root.MPEFit.cols.zenith[:]]
        
       
        ww = CorsikaWeighter(d,nfiles=1)
        print("W",weights)
        print("X",ww.get_weights(flux))

        #print('X',www,ww)
        www+=ww

    
    else:
        weights = np.full(len(mf),1./livetime)
        data_rate = weights[mf].sum()
        hist(d.root.MPEFit.cols.zenith[:][mf],range=[0,np.pi],w=weights[mf],logx=False,label=l).plot('err',logy=False)
    #hist(d.root.Homogenized_QTot.cols.value[:][mf],range=qtot_r,w=weights[mf],logx=True,label=l).plot(logy=True)
    #

w=flux(e,p)/g(e,p)/(1600./1200)/(800./600)

WWW = www.get_weights(flux)

print('WW',w)
print('XX',WWW)


print("MC rate",w[m].sum())
print("MC rate",WWW[m].sum())
print("DATA rate",data_rate)


print (WWW)
print (qtot_r)
print (m)

#hist(q[m],w=WWW[m],range=qtot_r,logx=True,label="Sum MC").plot(logy=True)
hist(z[m],w=WWW[m],range=[0,np.pi],logx=False,label="Sum MC").plot('err',logy=False)    
plt.legend()

plt.show()
