import numpy as np
import warnings
from .utils import has_table, get_table

class Null:
    """
    An identity object, useful as a starting point for accumulators, e.g.::

    total = Null()
    for i in range(10):
        total += SomeClassThatImplementsAddition(i)
    """
    def __iadd__(self, other):
        return other
    def __eq__(self, other):
        return isinstance(other, Null) or 0 == other

def append_dicts(first,second):
    if not first:
        return second
    if not second:
        return first    
    out={}
    assert(first.keys()==second.keys())
    for k in first.keys():
        out[k]=np.r_[first[k],second[k]]
    return out

def check_run_counts(table,nfiles):
    runs,counts = np.unique(table.cols.Run[:],return_counts=True)
    #more sophisticated checks go here?
    if len(runs)==nfiles:
        s = 'OK'
        ret = True
    else:
        s = 'Fail'
        ret = False 
    print ("Claimed Runs = {}, Found Runs = {}, {}".format(nfiles,len(runs),s))
    return ret

class Weighter:
    def __init__(self, surface, event_data, flux_map):
        self.surface =  surface
        self.event_data = event_data
        self.flux_map = flux_map

        l=None
        for v in self.event_data.values():
            if l is None:
                l = len(v)
            else:
                assert l==len(v)

    def get_weights(self,flux):
        
        surface = self.surface.get_extended_pdf(
            particle_type=self.event_data['type'],
            energy=self.event_data['energy'],
            cos_zen=self.event_data['cos_zenith'])
        
        mask = surface > 0
        if not np.all(mask):
            warnings.warn('simweights :: {} events out of {} were found to be outside the generation surface'
                          .format(np.logical_not(mask).sum(),mask.size))

        flux_params = { k : self.event_data[v][mask] for k, v in self.flux_map.items()}        
        event_weight = self.event_data['weight'][mask]      
        w = np.zeros_like(surface)
        w[mask] = event_weight * flux(**flux_params) / surface[mask] 
        return w

    def is_null(self):
        return (isinstance(self.surface,Null)
                and not self.event_data
                and not self.flux_map
                and not self.surface_map
                and not self.weight_name)

    def is_compatable(self,other):
        if self.is_null() or other.is_null():
            return True
        return ( self.event_data.keys()==other.event_data.keys()
                 and self.flux_map == other.flux_map)

    def __add__(self,other):
        assert self.is_compatable(other)

        if self.is_null():
            return other
        if other.is_null():
            return self

        surface = self.surface + other.surface
        event_data = append_dicts(self.event_data,other.event_data)
        return Weighter(surface, event_data, self.flux_map)
        
def make_weighter(info_obj,weight_obj,surface_func, surface_from_file,
                  event_data_func, flux_map):

    def _weighter(infile, sframe=True, nfiles=None):
        weight_table = get_table(infile, weight_obj)

        if sframe and has_table(infile, info_obj):
            info_table = get_table(infile, info_obj)

            surface = Null()
            for row in info_table:
                d = { x:row[x] for x in info_table.dtype.names }
                surface += surface_func(**d)
        else:
            infos = surface_from_file(infile)
            assert(nfiles is not None)
            
            surface=Null()
            for i in infos:
                surface += surface_func(**i)
            surface *= nfiles
            
        if nfiles is not None:
            check_run_counts(weight_table,nfiles)

        event_data=event_data_func(weight_table)
        return Weighter(surface,event_data,flux_map)

    return _weighter
