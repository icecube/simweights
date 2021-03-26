from copy import deepcopy
import numpy as np
from .fluxes import PDGCode

class GenerationSurface:
    def __init__(self, particle_type, nevents, spectrum, surface):
        self.particle_type = particle_type
        try:
            self.particle_name = PDGCode(particle_type).name
        except ValueError:
            self.particle_name = str(particle_type)
        self.nevents = nevents
        self.spectrum = deepcopy(spectrum)
        self.surface = deepcopy(surface)
        
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

    def get_energy_range(self, ptype):
        assert ptype == self.particle_type
        return self.spectrum.a, self.spectrum.b

    def __eq__(self,other):
        return self.is_compatible(other) and self.nevents == other.nevents
    
    def __add__(self, other):
        if isinstance(other, type(self)):
            if self.is_compatible(other):
                r = deepcopy(self)
                r.nevents = self.nevents + other.nevents
                return r
            else:
                return GenerationSurfaceCollection(self, other)
        else:
            raise TypeError("Can't add %s to %s" % (type(other).__name__, type(self).__name__))

    def __imul__(self,factor):
        self.nevents *= factor
        return self

    def __mul__(self, factor):
        s = deepcopy(self)
        s.__imul__(factor)
        return s

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __repr__(self):
        return "{}({}, {:7.3e}, {}, {})".format(
            self.__class__.__name__, self.particle_name,
            self.nevents, self.spectrum, self.surface)

class GenerationSurfaceCollection:
    """
    A collection of generation spectra, possibly for different particle types.
    """
    def __init__(self, *spectra):
        """
        :param spectra: a collection of GenerationProbabilities.
        """
        self.spectra = {}
        for s in spectra:
            self._insert(s)

    def _insert(self,surface):
        assert type(surface) == GenerationSurface
        key = int(surface.particle_type)
        if key not in self.spectra:
            self.spectra[key] = []

        for i,s in enumerate(self.spectra[key]):
            if surface.is_compatible(s):
                self.spectra[key][i] = s + surface
                break
        else:
            self.spectra[key].append(deepcopy(surface))

    def __add__(self,other):
        output = deepcopy(self)
        if isinstance(other, GenerationSurface):
            output._insert(other)
        elif isinstance(other, GenerationSurfaceCollection):
            for pt, ospectra in other.spectra.items():
                for ospec in ospectra:
                    output._insert(ospec)
        else:
            raise ValueError("Cannot add {} to {}".format(type(self),type(self)))
        return output

    def __mul__(self, factor):
        s = deepcopy(self)
        for p in s.spectra.values():
            for i in range(len(p)):
                p[i] *= factor
        return s

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def get_extended_pdf(self, particle_type, energy, cos_zen):
        energy = np.asarray(energy)
        cos_zen = np.asarray(cos_zen)
        count = np.zeros_like(energy, dtype=float)

        for ptype in np.unique(particle_type):
            mask = (particle_type == ptype)
            if np.any(mask):
                Em = energy[mask]
                ctm = cos_zen[mask]
                count[mask] += sum(p.get_extended_pdf(ptype, Em, ctm) for p in self.spectra[ptype])
        return count

    def get_energy_range(self, ptype):
        assert ptype in self.spectra
        assert len(self.spectra[ptype])
        emin = np.inf
        emax = -np.inf
        for p in self.spectra[ptype]:
            emin = min(emin,p.spectrum.a)
            emax = max(emax,p.spectrum.b)
        assert(np.isfinite(emin))
        assert(np.isfinite(emax))
        return emin, emax

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
        return (self.__class__.__name__+'('+
                ','.join(repr(y) for x in self.spectra.values() for y in x)+')')

    def __str__(self):
        s=[]
        for p,d in self.spectra.items():
            collections = []            
            for x in d:
                collections.append('N={:8.4g} {} {}'.format(x.nevents, x.spectrum, x.surface))
            s.append('     {:11} : '.format(x.particle_name)+
                     '\n                   '.join(collections))
        return '< '+self.__class__.__name__ + '\n'+ '\n'.join(s) + '\n>'
        
            
    
