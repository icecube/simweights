import warnings
from copy import copy

import numpy as np

from .fluxes import CosmicRayFlux
from .utils import get_column, get_table


class Weighter:
    """
    Abstract base class from which all weighers derive.

    Weighters will take a file object as input and calculate the weights of the events in the file
    for a given flux. As well as helper functions for all columns in the file. Weighters will keep
    track of the generation surface for the Monte Carlo in question. Weighters can be
    added together to form samples with differnt simulation parameters
    """

    def __init__(self, data, surface, event_map):
        self.data = data
        self.surface = surface
        self.event_map = event_map

    def get_column(self, table: str, column: str):
        """
        Helper function to get a specific column from the file
        """
        retval = []
        for datafile in self.data:
            retval = np.append(retval, get_column(get_table(datafile, table), column))
        return retval

    def get_weights(self, flux):
        """
        Calculate the weights for the sample in the weighter function.

        Multiplies the flux by the event weight and devides by the surface for every event in the
        Monte Carlo sample.
        """
        event_col = dict(
            energy=self.get_column(*self.event_map["energy"]),
            pdgid=self.get_column(*self.event_map["pdgid"]),
            cos_zen=np.cos(self.get_column(*self.event_map["zenith"])),
        )
        epdf = self.surface.get_epdf(**event_col)

        if isinstance(flux, CosmicRayFlux):
            flux_val = flux(event_col["energy"], event_col["pdgid"])
        else:
            flux_val = flux(**event_col)

        if self.event_map["event_weight"] is None:
            event_weight = np.ones_like(epdf)
        else:
            event_weight = self.get_column(*self.event_map["event_weight"])

        # Getting events with epdf=0 indicates some sort of mismatch between the
        # the surface and the dataset that can't be solved here so print a
        # warning and ignore the events
        mask = epdf > 0

        if not np.all(mask):
            warnings.warn(
                "simweights :: {} events out of {} were found to be "
                "outside the generation surface".format(np.logical_not(mask).sum(), mask.size)
            )
        weights = np.zeros_like(epdf)
        weights[mask] = (event_weight * flux_val)[mask] / epdf[mask]

        return weights

    def effective_area(self, pdgid=None, energy_bins=None, cos_zenith_bins=None):

        """
        Calculate The effective area for a given energy and zenith bins.

        Args:
            pdgid (PGDCode): The particle to calculate Effective area for
            energy_bins(array_like): an length N+1 array of energy bin edges
            coz_zenith_bins(array_like): an length M+1 array of energy bin edges

        Returns:
            array_like: An NxM array of effective areas

        """

        if energy_bins is None:
            energy_bins = self.surface.get_energy_range(pdgid)

        if cos_zenith_bins is None:
            cos_zenith_bins = self.surface.get_cos_zenith_range(pdgid)

        energy_bins = np.array(energy_bins)
        cos_zenith_bins = np.array(cos_zenith_bins)

        assert energy_bins.ndim == 1
        assert cos_zenith_bins.ndim == 1
        assert len(energy_bins) >= 2
        assert len(cos_zenith_bins) >= 2

        pdgid_col = self.get_column(*self.event_map["pdgid"])
        energy = self.get_column(*self.event_map["energy"])
        cos_zen = np.cos(self.get_column(*self.event_map["zenith"]))

        if pdgid is None:
            mask = np.ones_like(pdgid_col, dtype=bool)
            nspecies = len(self.surface.get_pdgids())
        else:
            mask = pdgid == pdgid_col
            nspecies = 1

        weights = self.get_weights(lambda energy, pdgid, cos_zen: 1)
        hist_val, czbin, enbin = np.histogram2d(
            cos_zen[mask],
            energy[mask],
            weights=weights[mask],
            bins=[cos_zenith_bins, energy_bins],
        )

        assert np.array_equal(enbin, energy_bins)
        assert np.array_equal(czbin, cos_zenith_bins)
        e_width, z_width = np.meshgrid(np.ediff1d(enbin), np.ediff1d(czbin))
        return hist_val / (e_width * 2 * np.pi * z_width * nspecies)

    def __add__(self, other):
        if type(self) is not type(other):
            raise ValueError("Cannot add {} to {}".format(type(self), type(self)))
        ret = copy(self)
        ret.surface += other.surface
        ret.data = self.data + other.data
        return ret
