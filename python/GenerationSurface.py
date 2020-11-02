from . import PDGCode
from .GenerationSurfaceCollection import GenerationSurfaceCollection
import copy

class GenerationSurface:
    def __init__(self,particle_type,nevents,spectrum,surface):
        self.particle_type=particle_type
        self.particle_name=PDGCode(self.particle_type).name
        self.nevents=nevents        
        self.spectrum=spectrum        
        self.surface=surface
        self.enorm = self.spectrum.total_integral()
        
        
    def __call__(self,energy,cos_zen):        
        return (self.nevents * self.spectrum.eval(energy)
                /self.spectrum.total_integral() /self.surface.area_weight(cos_zen))

    def __imul__(self, factor):
        self.nevents *= factor
        return self

    def __eq__(self,other):
        return self.is_compatible(other) and self.nevents==other.nevents
        
    def is_compatible(self, other):
        return (isinstance(other, type(self)) and
                self.particle_type == other.particle_type and
                self.spectrum==other.spectrum and
                self.surface ==other.surface)
    
    def __iadd__(self, other):
        if isinstance(other, type(self)):
            if self.is_compatible(other):
                self.nevents += other.nevents
                return self
            else:
                return GenerationSurfaceCollection([copy.deepcopy(self), other])
        else:
            raise TypeError("Can't add a %s to this %s" % (type(other).__name__, type(self).__name__))
 
    def __repr__(self):
        return "{}({}, {:7.3e}, {}, {})".format(
            self.__class__.__name__, self.particle_name,
            self.nevents, self.spectrum, self.surface)
    
