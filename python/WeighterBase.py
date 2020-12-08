import numpy as np

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
    def __init__(self,surface,event_data,flux_map):
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

        flux_params = { k:self.event_data[v] for k,v in self.flux_map.items()}
        surface = self.surface(particle_type=self.event_data['type'],
                               energy=self.event_data['energy'],
                               cos_zen=self.event_data['cos_zenith'])
        event_weight = self.event_data['weight']
        w = event_weight * flux(**flux_params) / surface
        #this shouldn't be here but corsika keeps decreacing the energy of primaries for some reason
        w[w==np.inf]=0
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
                 and self.flux_map==other.flux_map
                 and self.surface_map==other.surface_map
                 and self.weight_name == other.weight_name)
                 
    def __iadd__(self,other):
        assert self.is_compatable(other)

        if self.is_null():
            return other
        if other.is_null():
            return self

        self.surface+=other.surface
        self.event_data=append_dicts(self.event_data,other.event_data)
        return self
        
    
def NullWeighter():
    return Weighter(Null(),{},{},{},{})

def make_weighter(info_obj,weight_obj,surface_func, surface_from_weight_table,
                  event_data_func, flux_map):

    def _weighter(infile,sframe=True,nfiles=None):

        weight_table = getattr(infile.root,weight_obj)
    
        if sframe and hasattr(infile.root,info_obj):
            info_table = getattr(infile.root,info_obj)
            surface=Null()
            for r in info_table.iterrows():
                d = {x:r[x] for x in info_table.colnames}
                surface += surface_func(**d)       
        else:
            assert(nfiles is not None)
            infos = surface_from_weight_table(weight_table)
            surface=Null()
            for i in infos:
                surface += surface_func(**i)
            surface *= nfiles
            
        if nfiles is not None:
            check_run_counts(weight_table,nfiles)

        event_data=event_data_func(weight_table)
        return Weighter(surface,event_data,flux_map)

    return _weighter
