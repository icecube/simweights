import numpy as np
import inspect
from . import Null,UprightCylinder,PowerLaw,GenerationSurface,GenerationSurfaceCollection,ZenithBias
from pprint import pprint
from .utils import get_constant_column

class WeighterBase:
    def __init__(self,infile,sframe=True,nfiles=None):        
        self.weight_table = getattr(infile.root,self._weight_obj)        
        if sframe and hasattr(infile.root,self._info_obj) :
            self.info_table = getattr(infile.root,self._info_obj)
            self.gen = self.generator_from_info(self.info_table,self.get_gen)
            self.nfiles = len(np.unique(self.info_table.cols.run_id[:]))
            if nfiles is not None:
                assert self.nfiles == nfiles
        else:
            self.info_table=None
            if nfiles is None:            
                raise Exception("n_files not set")
            self.gen = self.gen_from_weight_obj(self.weight_table)
            self.gen *= nfiles
            self.nfiles = nfiles
            if not self.check_run_counts(infile):
                raise Exception("number of runs is probably wrong")
        self.flux_params = self.get_flux_params()
        self.surface_params = self.get_surface_params()
        self.p_int = self.get_interaction_prob()

    def get_interaction_prob(self):
        return 1.

    def check_run_counts(self,infile):
        table = getattr(infile.root,self._weight_obj)
        runs,counts = np.unique(table.cols.Run[:],return_counts=True)
        #more sophisticated checks go here
        if len(runs)==self.nfiles:
            s = 'OK'
            ret = True
        else:
            s = 'Fail'
            ret = False 
        print ("Claimed Runs = {}, Found Runs = {}, {}".format(self.nfiles,len(runs),s))
        return ret
        
    @staticmethod
    def generator_from_info(info_table,generator_function):
        gen=Null()
        for r in info_table.iterrows():
            d = {x:r[x] for x in info_table.colnames}
            gen += generator_function(**d)       
        return gen

    def get_weights(self,flux):

        f = flux(**self.flux_params)
        g = self.gen(**self.surface_params)

        weight = self.p_int * f / self._unit  / g        
        return weight
