import warnings

import numpy as np

from .utils import get_column, get_table


class Weighter:
    def __init__(self):
        raise NotImplementedError()

    def get_column(self, table: str, column: str):
        return np.ravel([get_column(get_table(d, table), column) for d in self.data])

    def get_weights(self, flux):
        epdf = self.surface.get_extended_pdf(**self.get_surface_params())
        flux_val = flux(**self.get_flux_params())
        event_weight = self.get_event_weight()

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

    def __add__(self, other):
        if type(self) is not type(self):
            raise ValueError("Cannot add {} to {}".format(type(self), type(self)))
        self.surface += other.surface
        self.data += other.data
        return self
