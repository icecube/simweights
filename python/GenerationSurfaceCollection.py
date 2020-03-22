import numpy as np

class GenerationSurfaceCollection:
    """
    A collection of generation spectra, possibly for different particle types.
    """
    def __init__(self, spectra):
        """
        :param spectra: a collection of GenerationProbabilities.
        """
        #from collections import defaultdict
        self.spectra = {}#defaultdict(list)
        for dist in spectra:
            key = int(dist.particle_type)
            if key not in self.spectra:
                self.spectra[key]=[]
            self.spectra[key].append(dist)
            
    def __call__(self, particle_type, energy, cos_zen):
        energy = np.asarray(energy)
        cos_zen = np.asarray(cos_zen)
        count = np.zeros_like(energy)
        for ptype in np.unique(particle_type):
            mask = particle_type==ptype
            if np.any(mask):
                Em = energy[mask]
                ctm = cos_zen[mask]
                count[mask] += sum(p(Em, ctm) for p in self.spectra[ptype])
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
