from copy import deepcopy

import numpy as np

from .pdgcode import PDGCode


class GenerationSurface:
    """
    This is a booking class which represents the surface in which Monte Carlo simulation is generated on.

    In IceCube the surface is an combines an energy, an area, and a solid angle as well as the number of
    events generated. This uses composision to store the energy spectrum in energy_dist and the area and
    solid angle stored in the surface.
    """

    def __init__(self, pdgid, energy_dist, spatial_dist):
        self.pdgid = pdgid
        try:
            self.particle_name = PDGCode(pdgid).name
        except ValueError:
            self.particle_name = str(pdgid)
        self.nevents = 1
        self.energy_dist = deepcopy(energy_dist)
        self.spatial_dist = deepcopy(spatial_dist)

    def get_epdf(self, pdgid, energy, cos_zen):
        """
        Get the extended pdf of an event.

        The pdf is the probability that an event with these parameters is generated. The pdf is multiplied
        by the number of events.
        """
        assert np.all(pdgid == self.pdgid)
        return self.nevents * self.energy_dist.pdf(energy) * self.spatial_dist.pdf(cos_zen)

    def get_surface_area(self):
        """
        Get the surface area in E * sr * m^2
        """
        return self.energy_dist.span * self.spatial_dist.etendue

    def is_compatible(self, other):
        """
        Returns if other class can be combined with the this class
        """
        return (
            isinstance(other, type(self))
            and self.pdgid == other.pdgid
            and self.energy_dist == other.energy_dist
            and self.spatial_dist == other.spatial_dist
        )

    def get_pdgids(self):
        """
        Return a list of pdgids that this surface represents
        """
        return [self.pdgid]

    def get_energy_range(self, pdgid):
        """
        Return the energy range for given particle type over all surfaces
        """
        assert (pdgid is None) or (pdgid == self.pdgid)
        return self.energy_dist.a, self.energy_dist.b

    def get_cos_zenith_range(self, pdgid):
        """
        Return the cos_zenith range for given particle type over all surfaces
        """
        assert (pdgid is None) or pdgid == self.pdgid
        return self.spatial_dist.cos_zen_min, self.spatial_dist.cos_zen_max

    def __eq__(self, other):
        return self.is_compatible(other) and self.nevents == other.nevents

    def __add__(self, other):
        if isinstance(other, GenerationSurface):
            if self.is_compatible(other):
                new_surface = deepcopy(self)
                new_surface.nevents = self.nevents + other.nevents
                return new_surface
            return GenerationSurfaceCollection(self, other)
        raise TypeError("Can't add %s to %s" % (type(other).__name__, type(self).__name__))

    def __imul__(self, factor):
        self.nevents *= factor
        return self

    def __mul__(self, factor):
        new_surface = deepcopy(self)
        new_surface.__imul__(factor)
        return new_surface

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __repr__(self):
        return "{} * {}({}, {}, {})".format(
            self.nevents, self.__class__.__name__, self.particle_name, self.energy_dist, self.spatial_dist
        )


class GenerationSurfaceCollection:
    """
    A collection of generation spectra, possibly for different particle types.
    """

    def __init__(self, *spectra):
        """
        :param spectra: a collection of GenerationProbabilities.
        """
        self.spectra = {}
        for spec in spectra:
            self._insert(spec)

    def _insert(self, surface):
        assert isinstance(surface, GenerationSurface)
        key = int(surface.pdgid)
        if key not in self.spectra:
            self.spectra[key] = []

        for i, spec in enumerate(self.spectra[key]):
            if surface.is_compatible(spec):
                self.spectra[key][i] = spec + surface
                break
        else:
            self.spectra[key].append(deepcopy(surface))

    def __add__(self, other):
        output = deepcopy(self)
        if isinstance(other, GenerationSurface):
            output._insert(other)
        elif isinstance(other, GenerationSurfaceCollection):
            for _, ospectra in other.spectra.items():
                for ospec in ospectra:
                    output._insert(ospec)
        else:
            raise ValueError("Cannot add {} to {}".format(type(self), type(self)))
        return output

    def __mul__(self, factor):
        new_surface = deepcopy(self)
        for subsurf in new_surface.spectra.values():
            for i, _ in enumerate(subsurf):
                subsurf[i] *= factor
        return new_surface

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def get_epdf(self, pdgid, energy, cos_zen):
        """
        Get the extended pdf of an event.

        The pdf is the probability that an event with these parameters is generated. The pdf is multiplied
        by the number of events.
        """
        energy = np.asarray(energy)
        cos_zen = np.asarray(cos_zen)
        count = np.zeros_like(energy, dtype=float)

        for ptype in np.unique(pdgid):
            mask = ptype == pdgid
            if np.any(mask):
                masked_energy = energy[mask]
                masked_cos_zen = cos_zen[mask]
                count[mask] += sum(
                    p.get_epdf(ptype, masked_energy, masked_cos_zen) for p in self.spectra[ptype]
                )
        return count

    def get_pdgids(self):
        """
        Return a list of pdgids that this surface represents
        """
        return sorted(self.spectra.keys())

    def get_energy_range(self, pdgid):
        """
        Return the energy range for given particle type over all surfaces
        """
        if pdgid is None:
            pdgids = sorted(self.spectra.keys())
        else:
            pdgids = [pdgid]

        assert set(pdgids).issubset(self.spectra.keys())
        emin = np.inf
        emax = -np.inf
        for pid in pdgids:
            for surf in self.spectra[pid]:
                emin = min(emin, surf.energy_dist.a)
                emax = max(emax, surf.energy_dist.b)
        assert np.isfinite(emin)
        assert np.isfinite(emax)
        return emin, emax

    def get_cos_zenith_range(self, pdgid):
        """
        Return the cos zenith range for given particle type over all surfaces
        """

        if pdgid is None:
            pdgids = sorted(self.spectra.keys())
        else:
            pdgids = [pdgid]

        assert set(pdgids).issubset(self.spectra.keys())
        czmin = np.inf
        czmax = -np.inf
        for pid in pdgids:
            for surf in self.spectra[pid]:
                czmin = min(czmin, surf.spatial_dist.cos_zen_min)
                czmax = max(czmax, surf.spatial_dist.cos_zen_max)
        assert np.isfinite(czmin)
        assert np.isfinite(czmax)
        return czmin, czmax

    def __eq__(self, other):
        # must handle the same set of particle types
        if set(self.spectra.keys()) != set(other.spectra.keys()):
            return False
        for k in self.spectra:
            spec1 = self.spectra[k]
            spec2 = other.spectra[k]
            # must have the same number of unique spectra
            if len(spec1) != len(spec2):
                return False
            # exactly one match for each energy_dist
            for subspec1 in spec1:
                if sum(subspec1 == subspec2 for subspec2 in spec2) != 1:
                    return False
        return True

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ",".join(repr(y) for x in self.spectra.values() for y in x)
            + ")"
        )

    def __str__(self):
        outstrs = []
        for specs in self.spectra.values():
            collections = []
            for subspec in specs:
                collections.append(
                    "N={:8.4g} {} {}".format(subspec.nevents, subspec.energy_dist, subspec.spatial_dist)
                )
            outstrs.append(
                "     {:11} : ".format(specs[0].particle_name) + "\n                   ".join(collections)
            )
        return "< " + self.__class__.__name__ + "\n" + "\n".join(outstrs) + "\n>"
