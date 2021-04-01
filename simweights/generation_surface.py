from copy import deepcopy

import numpy as np

from .fluxes import PDGCode


class GenerationSurface:
    """
    This is a booking class which represents the surface in which Monte Carlo simulation is generated on.

    In IceCube the surface is an combines an energy, an area, and a solid angle as well as the number of
    events generated. This uses composision to store the energy spectrum in spectrum and the area and solid
    angle stored in the surface.
    """

    def __init__(self, particle_type, nevents, spectrum, surface):
        self.particle_type = particle_type
        try:
            self.particle_name = PDGCode(particle_type).name
        except ValueError:
            self.particle_name = str(particle_type)
        self.nevents = nevents
        self.spectrum = deepcopy(spectrum)
        self.surface = deepcopy(surface)

    def get_epdf(self, particle_type, energy, cos_zen):
        """
        Get the extended pdf of an event.

        The pdf is the probability that an event with these parameters is generated. The pdf is multiplied
        by the number of events.
        """
        assert np.all(particle_type == self.particle_type)
        return self.nevents * self.spectrum.pdf(energy) * self.surface.pdf(cos_zen)

    def get_surface_area(self):
        """
        Get the surface area in E * sr * m^2
        """
        return self.spectrum.span * self.surface.etendue

    def is_compatible(self, other):
        """
        Returns if other class can be combined with the this class
        """
        return (
            isinstance(other, type(self))
            and self.particle_type == other.particle_type
            and self.spectrum == other.spectrum
            and self.surface == other.surface
        )

    def get_energy_range(self, ptype):
        """
        Return the energy range for given particle type over all surfaces
        """
        assert ptype == self.particle_type
        return self.spectrum.a, self.spectrum.b

    def __eq__(self, other):
        return self.is_compatible(other) and self.nevents == other.nevents

    def __add__(self, other):
        if isinstance(other, type(self)):
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
        return "{}({}, {:7.3e}, {}, {})".format(
            self.__class__.__name__, self.particle_name, self.nevents, self.spectrum, self.surface
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
        key = int(surface.particle_type)
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

    def get_epdf(self, particle_type, energy, cos_zen):
        """
        Get the extended pdf of an event.

        The pdf is the probability that an event with these parameters is generated. The pdf is multiplied
        by the number of events.
        """
        energy = np.asarray(energy)
        cos_zen = np.asarray(cos_zen)
        count = np.zeros_like(energy, dtype=float)

        for ptype in np.unique(particle_type):
            mask = particle_type == ptype
            if np.any(mask):
                masked_energy = energy[mask]
                masked_cos_zen = cos_zen[mask]
                count[mask] += sum(
                    p.get_epdf(ptype, masked_energy, masked_cos_zen) for p in self.spectra[ptype]
                )
        return count

    def get_energy_range(self, ptype):
        """
        Return the energy range for given particle type over all surfaces
        """
        assert ptype in self.spectra
        assert len(self.spectra[ptype])
        emin = np.inf
        emax = -np.inf
        for surf in self.spectra[ptype]:
            emin = min(emin, surf.spectrum.a)
            emax = max(emax, surf.spectrum.b)
        assert np.isfinite(emin)
        assert np.isfinite(emax)
        return emin, emax

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
            # exactly one match for each spectrum
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
                    "N={:8.4g} {} {}".format(subspec.nevents, subspec.spectrum, subspec.surface)
                )
            outstrs.append(
                "     {:11} : ".format(specs[0].particle_name) + "\n                   ".join(collections)
            )
        return "< " + self.__class__.__name__ + "\n" + "\n".join(outstrs) + "\n>"
