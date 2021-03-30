import numpy as np
import warnings
from .utils import has_table, get_table, get_column

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
    def __init__(self, surface, data):        
        self.surface = surface
        self.data = data
                
    def get_column(self, table:str, column:str):
        return np.r_[[get_column(get_table(d,table),column) for d in self.data]]

    def get_weights(self, flux):
        epdf = self.surface.get_extended_pdf(**self.get_surface_params())
        flux_val = flux(**self.get_flux_params())
        event_weight = self.get_event_weight()

        #Getting events with epdf=0 indicates some sort of mismatch between the
        #the surface and the dataset that can't be solved here so print a
        #warning and ignore the events
        mask = epdf > 0
        if not np.all(mask):
            warnings.warn('simweights :: {} events out of {} were found to be outside the generation surface'
                          .format(np.logical_not(mask).sum(),mask.size))
       
        w = np.zeros_like(epdf)
        w[mask] = (event_weight * flux_val)[mask] / epdf[mask] 
        return w

    def __add__(self,other):
        if type(self) is not type(self):
            raise ValueError("Cannot add {} to {}".format(type(self),type(self)))        
        self.surface+=other.surface
        self.data+=other.data
        return self
        
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
        return Weighter(surface, [infile])

    return _weighter
