# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from simweights._pdgcode import PDGCode
from simweights._powerlaw import PowerLaw
from simweights._spatial import SpatialDist

if TYPE_CHECKING:
    from collections.abc import Mapping


class GenerationSurface:
    """Represents a surface on which Monte Carlo simulation was generated on.

    The number of events thrown, the spatial distribution, and the energy distribution are stored in this
    class. Each particle type is stored separately.
    """

    def __init__(self, pdgid: "PDGCode | int", nevents: float, power_law: PowerLaw, spatial: SpatialDist) -> None:
        self.pdgid: PDGCode = PDGCode(pdgid)
        self.nevents = nevents
        self.power_law = power_law
        self.spatial = spatial

    def equivalent(self, surface: Any) -> bool:
        """Test for weather two surfaces cand be combined into a single surface with the sum of the nevents."""
        if not isinstance(surface, self.__class__):
            return False
        return self.pdgid == surface.pdgid and self.power_law == surface.power_law and self.spatial == surface.spatial

    def scale(self, factor: float) -> None:
        """Scale the number of events by this factor."""
        self.nevents *= factor

    def __eq__(self, surface: object) -> bool:
        if not isinstance(surface, self.__class__):
            return False
        return self.equivalent(surface) and self.nevents == surface.nevents

    def get_energy_range(self, pdgid: "PDGCode | None" = None) -> "tuple[float, float]":
        """Return the energy range for given particle type over all surfaces."""
        if pdgid is None:
            pdgid = self.pdgid
        assert pdgid == self.pdgid
        return self.power_law.a, self.power_law.b

    def get_cos_zenith_range(self, pdgid: "PDGCode | None" = None) -> "tuple[float, float]":
        """Return the cos zenith range for given particle type over all surfaces."""
        if pdgid is None:
            pdgid = self.pdgid
        assert pdgid == self.pdgid
        return self.spatial.cos_zen_min, self.spatial.cos_zen_max

    def get_epdf(self, weight_cols: "Mapping[str, NDArray[np.float64]]") -> NDArray[np.float64]:
        """Get the extended pdf of sample.

        The pdf is the probability that an event with these parameters is generated. The pdf is multiplied
        by the number of events.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pdgid.name}, {self.nevents}, {self.power_law}, {self.spatial})"


class CompositeSurface:
    """Represents two or more surface on which Monte Carlo simulation was generated on.

    This combines multiple surfaces either because the surfaces had different parameters
    or because they are different particle types
    """

    def __init__(self, *args: GenerationSurface) -> None:
        self.components: dict[PDGCode, list[GenerationSurface]] = {}
        for a in args:
            self.insert(a)

    def insert(self, new_surface: Any) -> None:
        """Insert a new surface into this composite service."""
        if isinstance(new_surface, CompositeSurface):
            for val in new_surface.components.values():
                for v in val:
                    self.insert(v)
        elif isinstance(new_surface, GenerationSurface):
            key = new_surface.pdgid
            if key not in self.components:
                self.components[key] = []
            for comp in self.components[key]:
                if new_surface.equivalent(comp):
                    comp.nevents += new_surface.nevents
                    break
            else:
                self.components[key].append(deepcopy(new_surface))
        else:
            mesg = f"Cannot combine {type(self)} to {type(new_surface)}"
            raise TypeError(mesg)

    def scale(self, factor: float) -> None:
        """Scale the number of events by this factor."""
        for val in self.components.values():
            for comp in val:
                comp.scale(factor)

    def get_epdf(self, weight_cols: "Mapping[str, ArrayLike]") -> NDArray[np.float64]:
        """Get the extended pdf of an event.

        The pdf is the probability that an event with these parameters is generated. The pdf is multiplied
        by the number of events.
        """
        cols = {}
        b = np.broadcast(*(v for v in weight_cols.values()))
        cols = {k: np.asarray(v, dtype=np.float64) for k, v in weight_cols.items()}
        count = np.zeros(b.shape, dtype=np.float64)
        # loop over particle type
        for ptype in np.unique(weight_cols["pdgid"]):
            mask = ptype == weight_cols["pdgid"]
            cc = {k: v[mask] for k, v in cols.items()}
            # loop over different datasets of the same particle type
            for surface in self.components[ptype]:
                count[mask] += surface.get_epdf(cc)
        return count

    def get_energy_range(self: "CompositeSurface", pdgid: "PDGCode | None" = None) -> "tuple[float, float]":
        """Return the energy range for given particle type over all surfaces."""
        pdgids = sorted(self.components.keys()) if pdgid is None else [pdgid]

        assert set(pdgids).issubset(self.components.keys())
        emin = np.inf
        emax = -np.inf
        for pid in pdgids:
            for surf in self.components[pid]:
                a, b = surf.get_energy_range()
                emin = min(emin, a)
                emax = max(emax, b)
        assert np.isfinite(emin)
        assert np.isfinite(emax)
        return emin, emax

    def get_cos_zenith_range(self: "CompositeSurface", pdgid: "PDGCode | None" = None) -> "tuple[float, float]":
        """Return the cos zenith range for given particle type over all surfaces."""
        pdgids = sorted(self.components.keys()) if pdgid is None else [pdgid]

        assert set(pdgids).issubset(self.components.keys())
        czmin = np.inf
        czmax = -np.inf
        for pid in pdgids:
            for surf in self.components[pid]:
                a, b = surf.get_cos_zenith_range()
                czmin = min(czmin, a)
                czmax = max(czmax, b)
        assert np.isfinite(czmin)
        assert np.isfinite(czmax)
        return czmin, czmax

    def __eq__(self, other: object) -> bool:
        # must handle the same set of particle types
        if isinstance(other, GenerationSurface):
            return self == CompositeSurface(other)
        if not isinstance(other, CompositeSurface):
            return False
        if set(self.components.keys()) != set(other.components.keys()):
            return False
        for pdgid, spec1 in self.components.items():
            spec2 = other.components[pdgid]
            # must have the same number of unique spectra
            if len(spec1) != len(spec2):
                return False
            # exactly one match for each energy_dist
            for subspec1 in spec1:
                if sum(subspec1 == subspec2 for subspec2 in spec2) != 1:
                    return False
        return True

    def __str__(self) -> str:
        outstrs = []
        for pdgid, specs in self.components.items():
            outstrs.append(f"     {pdgid.name:>11} : " + "\n                   ".join([str(s) for s in specs]))
        return "< " + self.__class__.__name__ + "\n" + "\n".join(outstrs) + "\n>"

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(\n  " + ",\n  ".join(repr(y) for x in self.components.values() for y in x) + ",\n)"
