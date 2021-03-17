import copy
import numpy as np
from .fluxes import PDGCode

class GenerationSurface:
    def __init__(self, particle_type, nevents, spectrum, surface):
        self.particle_type = particle_type
        self.particle_name = PDGCode(self.particle_type).name
        self.nevents = nevents
        self.spectrum = copy.deepcopy(spectrum)
        self.surface = copy.deepcopy(surface)
        
    def get_extended_pdf(self, particle_type, energy, cos_zen):
        assert(np.all(particle_type == self.particle_type))
        return self.nevents * self.spectrum.pdf(energy) * self.surface.pdf(cos_zen)

    def get_surface_area(self):
        return (self.spectrum.span*self.surface.etendue)
    
    def is_compatible(self, other):
        return (isinstance(other, type(self)) and
                self.particle_type == other.particle_type and
                self.spectrum == other.spectrum and
                self.surface == other.surface)

    def __eq__(self,other):
        return self.is_compatible(other) and self.nevents == other.nevents
    
    def __add__(self, other):
        if isinstance(other, type(self)):
            if self.is_compatible(other):
                self.nevents += other.nevents
                return self
            else:
                return GenerationSurfaceCollection([self, other])
        else:
            raise TypeError("Can't add a %s to this %s" % (type(other).__name__, type(self).__name__))

    def __imul__(self, factor):
        self.nevents *= factor
        return self
 
    def __repr__(self):
        return "{}({}, {:7.3e}, {}, {})".format(
            self.__class__.__name__, self.particle_name,
            self.nevents, self.spectrum, self.surface)

class GenerationSurfaceCollection:
    """
    A collection of generation spectra, possibly for different particle types.
    """
    def __init__(self, spectra):
        """
        :param spectra: a collection of GenerationProbabilities.
        """
        #from collections import defaultdict
        self.spectra = {}
        for dist in spectra:
            key = int(dist.particle_type)
            if key not in self.spectra:
                self.spectra[key] = []
            self.spectra[key].append(copy.deepcopy(dist))

    def get_extended_pdf(self, particle_type, energy, cos_zen):
        energy = np.asarray(energy)
        cos_zen = np.asarray(cos_zen)
        count = np.zeros_like(energy)

        for ptype in np.unique(particle_type):
            mask = (particle_type == ptype)
            if np.any(mask):
                Em = energy[mask]
                ctm = cos_zen[mask]
                count[mask] += sum(p.get_extended_pdf(ptype, Em, ctm) for p in self.spectra[ptype])
        return count

    def __imul__(self, factor):
        for spectra in self.spectra.values():
            for prob in spectra:
                prob *= factor
        return self

    def __idiv__(self, factor):
        self *= (1./factor)
        return self

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            for pt, ospectra in other.spectra.items():
                for ospec in ospectra:
                    for spec in self.spectra[pt]:
                        if spec.is_compatible(ospec):
                            spec += ospec
                            break
                    else:
                        self.spectra[pt].append(ospec)
            return self
        else:
            if other.particle_type in self.spectra:
                for spec in self.spectra[other.particle_type]:
                    if spec.is_compatible(other):
                        spec += other
                        break
                else:
                    self.spectra[other.particle_type].append(other)
            else:
                self.spectra[other.particle_type]=[other]
            return self

    def __eq__(self, other):
        # must handle the same set of particle types
        if set(self.spectra.keys()) != set(other.spectra.keys()):
            return False
        for k in self.spectra:
            s1 = self.spectra[k]
            s2 = other.spectra[k]
            # must have the same number of unique spectra
            if len(s1) != len(s2):
                return False
            # exactly one match for each spectrum
            for p1 in s1:
                if sum(p1 == p2 for p2 in s2) != 1:
                    return False
        return True

    def __repr__(self):
        return (self.__class__.__name__+'(['+
                ','.join(repr(x) for x in self.spectra.values())+'])')

    def __str__(self):
        s=[]
        for p,d in self.spectra.items():
            collections = []            
            for x in d:
                collections.append('N={:8.4g} {} {}'.format(x.nevents, x.spectrum, x.surface))
            s.append('     {:11} : '.format(x.particle_name)+
                     '                 \n'.join(collections))
        return '< '+self.__class__.__name__ + '\n'+ '\n'.join(s) + '\n>'
        
            
    
