import warnings

import numpy as np

from .cylinder import VolumeCorrCylinder
from .GenerationSurface import GenerationSurface
from .powerlaw import PowerLaw
from .utils import Null, get_column, get_table


class Weighter:
    def __init__(self, surface, data):
        self.surface = surface
        self.data = data

    def get_column(self, table: str, column: str):
        return np.ravel([get_column(get_table(d, table), column) for d in self.data])

    def get_weights(self, flux):
        epdf = self.surface.get_extended_pdf(**self._get_surface_params())
        flux_val = flux(**self._get_flux_params())
        event_weight = self._get_event_weight()

        # Getting events with epdf=0 indicates some sort of mismatch between the
        # the surface and the dataset that can't be solved here so print a
        # warning and ignore the events
        mask = epdf > 0
        if not np.all(mask):
            warnings.warn(
                "simweights :: {} events out of {} were found to be "
                "outside the generation surface".format(np.logical_not(mask).sum(), mask.size)
            )

        w = np.zeros_like(epdf)
        w[mask] = (event_weight * flux_val)[mask] / epdf[mask]
        return w

    @staticmethod
    def _get_surface(smap):
        assert smap["power_law_index"] < 0
        surface = VolumeCorrCylinder(
            smap["cylinder_height"],
            smap["cylinder_radius"],
            np.cos(smap["max_zenith"]),
            np.cos(smap["min_zenith"]),
        )
        spectrum = PowerLaw(smap["power_law_index"], smap["min_energy"], smap["max_energy"])
        return GenerationSurface(smap["primary_type"], smap["n_events"], spectrum, surface)

    def _get_surface_params(self):
        raise NotImplementedError()

    def _get_flux_params(self):
        raise NotImplementedError()

    def _get_event_weight(self):
        raise NotImplementedError()

    def __add__(self, other):
        if type(self) is not type(self):
            raise ValueError("Cannot add {} to {}".format(type(self), type(self)))
        self.surface += other.surface
        self.data += other.data
        return self


class MapWeighter(Weighter):
    def __init__(self, infile, nfiles):
        assert nfiles is not None
        surface = Null()
        for smap in self._get_surface_map(infile):
            surface += nfiles * self._get_surface(smap)
        super().__init__(surface, [infile])

    @staticmethod
    def _get_surface_map(infile):
        raise NotImplementedError()
