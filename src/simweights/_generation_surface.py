# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

from collections import namedtuple
from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._pdgcode import PDGCode
from ._powerlaw import PowerLaw
from ._spatial import SpatialDist

SurfaceTuple = namedtuple("SurfaceTuple", ["pdgid", "nevents", "energy_dist", "spatial_dist"])


class GenerationSurface:
    """Represents a surface on which Monte Carlo simulation was generated on.

    The number of events thrown, the spatial distribution, and the energy distribution are stored in this
    class. Each particle type is stored separately.
    """

    def __init__(self: GenerationSurface, *spectra: SurfaceTuple) -> None:
        """:param spectra: a collection of GenerationProbabilities."""
        self.spectra: dict[int, list[SurfaceTuple]] = {}
        for spec in spectra:
            self._insert(spec)

    def _insert(self: GenerationSurface, surface: SurfaceTuple) -> None:
        key = int(surface.pdgid)
        if key not in self.spectra:
            self.spectra[key] = []

        for i, spec in enumerate(self.spectra[key]):
            if surface.energy_dist == spec.energy_dist and surface.spatial_dist == spec.spatial_dist:
                self.spectra[key][i] = spec._replace(nevents=surface.nevents + spec.nevents)
                break
        else:
            self.spectra[key].append(deepcopy(surface))

    def __add__(self: GenerationSurface, other: int | GenerationSurface) -> GenerationSurface:
        output = deepcopy(self)
        if other == 0:
            return output
        if not isinstance(other, GenerationSurface):
            mesg = f"Cannot add {type(self)} to {type(other)}"
            raise TypeError(mesg)
        for ospectra in other.spectra.values():
            for ospec in ospectra:
                output._insert(ospec)
        return output

    def __radd__(self: GenerationSurface, other: int | GenerationSurface) -> GenerationSurface:
        return self + other

    def __mul__(self: GenerationSurface, factor: float) -> GenerationSurface:
        new_surface = deepcopy(self)
        for subsurf in new_surface.spectra.values():
            for i, _ in enumerate(subsurf):
                subsurf[i] = subsurf[i]._replace(nevents=factor * subsurf[i].nevents)
        return new_surface

    def __rmul__(self: GenerationSurface, factor: float) -> GenerationSurface:
        return self.__mul__(factor)

    def get_epdf(
        self: GenerationSurface,
        pdgid: ArrayLike,
        energy: ArrayLike,
        cos_zen: ArrayLike,
    ) -> NDArray[np.float64]:
        """Get the extended pdf of an event.

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
                    p.nevents * p.energy_dist.pdf(masked_energy) * p.spatial_dist.pdf(masked_cos_zen)
                    for p in self.spectra[ptype]
                )
        return count

    def get_pdgids(self: GenerationSurface) -> list[int | PDGCode]:
        """Return a list of pdgids that this surface represents."""
        return sorted(self.spectra.keys())

    def get_energy_range(self: GenerationSurface, pdgid: PDGCode | None) -> tuple[float, float]:
        """Return the energy range for given particle type over all surfaces."""
        pdgids = sorted(self.spectra.keys()) if pdgid is None else [pdgid]

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

    def get_cos_zenith_range(self: GenerationSurface, pdgid: PDGCode | None) -> tuple[float, float]:
        """Return the cos zenith range for given particle type over all surfaces."""
        pdgids = sorted(self.spectra.keys()) if pdgid is None else [pdgid]

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

    def __eq__(self: GenerationSurface, other: object) -> bool:
        # must handle the same set of particle types
        if not isinstance(other, GenerationSurface):
            return False
        if set(self.spectra.keys()) != set(other.spectra.keys()):
            return False
        for pdgid, spec1 in self.spectra.items():
            spec2 = other.spectra[pdgid]
            # must have the same number of unique spectra
            if len(spec1) != len(spec2):
                return False
            # exactly one match for each energy_dist
            for subspec1 in spec1:
                if sum(subspec1 == subspec2 for subspec2 in spec2) != 1:
                    return False
        return True

    def __repr__(self: GenerationSurface) -> str:
        return (
            self.__class__.__name__
            + "("
            + ",".join(repr(y) for x in self.spectra.values() for y in x)
            + ")"
        )

    def __str__(self: GenerationSurface) -> str:
        outstrs = []
        for pdgid, specs in self.spectra.items():
            try:
                ptype = PDGCode(pdgid).name
            except ValueError:  # noqa: PERF203
                ptype = str(pdgid)

            collections = [
                f"N={subspec.nevents} {subspec.energy_dist} {subspec.spatial_dist}" for subspec in specs
            ]
            outstrs.append(f"     {ptype:>11} : " + "\n                   ".join(collections))
        return "< " + self.__class__.__name__ + "\n" + "\n".join(outstrs) + "\n>"


def generation_surface(
    pdgid: int | PDGCode,
    energy_dist: PowerLaw,
    spatial_dist: SpatialDist,
) -> GenerationSurface:
    """Convenience function to generate a GenerationSurface for a single particle type."""
    return GenerationSurface(
        SurfaceTuple(pdgid=pdgid, nevents=1.0, energy_dist=energy_dist, spatial_dist=spatial_dist),
    )
