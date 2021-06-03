import inspect
import warnings

import numpy as np

from .utils import get_column, get_table


class Weighter:
    """
    Abstract base class from which all weighers derive.

    Weighters will take a file object as input and calculate the weights of the events in the file
    for a given flux. As well as helper functions for all columns in the file. Weighters will keep
    track of the generation surface for the Monte Carlo in question. Weighters can be
    added together to form samples with differnt simulation parameters
    """

    def __init__(self, data: list, surface, event_map: dict):
        self.data = list(data)
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
        Calculate the weights for the sample for the given flux.

        :param:`flux` can be any one of several types:

        * An instance of :py:class:`nuflux.FluxFunction` from `nuflux <https://github.com/icecube/nuflux>`_
        * An instance of :py:class:`simweights.CosmicRays`
        * A function with 2 to 3 parameters. If the function takes two parameter it will be pdgid and
          energy. if three it will be pdgid, energy, and cos_zenith.
        * An iterable the same length os the datasample. If you have other means of calculating the flux
          for each event than the above options this can be useful.
        * A scalar number. calculate the unrealisitc scenario where all events have the same flux, this can
          be useful for testing. If the value is 1 then the return value will be the well known quantity
          OneWeight
        """
        event_col = dict(
            energy=self.get_column(*self.event_map["energy"]),
            pdgid=self.get_column(*self.event_map["pdgid"]).astype(np.int32),
            cos_zen=np.cos(self.get_column(*self.event_map["zenith"])),
        )
        epdf = self.surface.get_epdf(**event_col)

        # calculate the flux based on which type of flux it is
        if hasattr(flux, "getFlux"):
            # this is a nuflux model
            assert callable(flux.getFlux)
            flux_val = 1e4 * flux.getFlux(event_col["pdgid"], event_col["energy"], event_col["cos_zen"])
        elif callable(flux):
            # this is a cosmic ray flux model or just a function
            arguments = {k: event_col[k] for k in inspect.signature(flux).parameters.keys()}
            flux_val = flux(**arguments)
        elif hasattr(flux, "__len__"):
            # this is an array with a length equal to the number of events
            flux_val = np.array(flux, dtype=float)
        elif np.isscalar(flux):
            # this is a scalar
            flux_val = np.full(epdf.shape, flux)
        else:
            raise ValueError("I do not understand what to do with flux {}".format(flux))
        assert flux_val.shape == epdf.shape

        # Some generators don't have an event weight so just let them use None
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
                "simweights :: {} events out of {} were found to be outside the generation surface."
                "This could indicate a problem with this dataset.".format(
                    np.logical_not(mask).sum(), mask.size
                )
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

        weights = self.get_weights(1)
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

    def __iadd__(self, other):
        if self.event_map != other.event_map:
            raise ValueError("Cannot add {} to {}".format(type(self), type(self)))
        self.data = [*self.data, *other.data]
        self.surface += other.surface
        return self

    def __add__(self, other):
        if self.event_map != other.event_map:
            raise ValueError("Cannot add {} to {}".format(type(self), type(self)))
        return Weighter([*self.data, *other.data], self.surface + other.surface, self.event_map)
