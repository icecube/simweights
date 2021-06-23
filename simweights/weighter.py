import inspect
import warnings
from pprint import pformat

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

    def __init__(self, data: list, surface):

        colnames = set([])
        for _, event_map in data:
            for colname in event_map.keys():
                colnames.add(colname)

        self.data = list(data)
        self.surface = surface
        self.colnames = list(colnames)

    def get_column(self, table: str, column: str):
        """
        Helper function to get a specific column from the file
        """
        retval = []
        for datafile, _ in self.data:
            retval = np.append(retval, get_column(get_table(datafile, table), column))
        return retval

    def get_weight_column(self, name: str):
        """
        Helper function to get a column needed in the weight calculation
        """
        retval = []
        for datafile, event_map in self.data:
            if event_map[name] is None:
                tablename, columnname = event_map["energy"]
                fileval = np.full(len(get_column(get_table(datafile, tablename), columnname)), 1)
            else:
                tablename, columnname = event_map[name]
                fileval = get_column(get_table(datafile, tablename), columnname)

            retval = np.append(retval, fileval)
        if name == "pdgid":
            retval = retval.astype(np.int32)
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
            energy=self.get_weight_column("energy"),
            pdgid=self.get_weight_column("pdgid").astype(np.int32),
            cos_zen=np.cos(self.get_weight_column("zenith")),
        )
        epdf = self.surface.get_epdf(**event_col)
        event_weight = self.get_weight_column("event_weight")

        # calculate the flux based on which type of flux it is
        if hasattr(flux, "getFlux"):
            # this is a nuflux model
            assert callable(flux.getFlux)
            flux_val = 1e4 * flux.getFlux(
                self.get_weight_column("pdgid"),
                self.get_weight_column("energy"),
                np.cos(self.get_weight_column("zenith")),
            )
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

    def effective_area(self, energy_bins, cos_zenith_bins, mask=None):
        """
        Calculate The effective area for the given energy and zenith bins.

        This is accomplished by histogramming the generation surface the simulation sample
        in energy and zenith bins and dividing by the size of the energy and solid angle of each bin.
        If mask is passed as a parameter, only events which are included in the mask are used.

        .. Note ::

            If the sample contains more than one type of primary particle, then the result will be
            averaged over the number of particles. This is usually what you want. However, it can
            cause strange behavior if there is a small number of one type. In this case, the mask
            should be used to select the particle types individually.

        Args:
            energy_bins(array_like): A length N+1 array of energy bin edges
            coz_zenith_bins(array_like): A length M+1 array of cos(zenith) bin edges
            mask(array_like): boolean array where 1 indicates to use the event in the calculation.
                Must have the same length as the simulation sample.

        Returns:
            array_like: An NxM array of effective areas. Where N is the number of energy bins and
                M is the number of cos(zenith) bins.

        """

        energy_bins = np.array(energy_bins)
        cos_zenith_bins = np.array(cos_zenith_bins)

        assert energy_bins.ndim == 1
        assert cos_zenith_bins.ndim == 1
        assert len(energy_bins) >= 2
        assert len(cos_zenith_bins) >= 2

        energy = self.get_weight_column("energy")
        cos_zen = np.cos(self.get_weight_column("zenith"))

        weights = self.get_weights(1)
        if mask is None:
            mask = np.full(weights.size, 1, dtype=bool)

        assert mask.dtype == bool
        assert mask.shape == weights.shape

        hist_val, czbin, enbin = np.histogram2d(
            cos_zen[mask],
            energy[mask],
            weights=weights[mask],
            bins=[cos_zenith_bins, energy_bins],
        )

        nspecies = len(np.unique(self.get_weight_column("pdgid")[mask]))

        assert np.array_equal(enbin, energy_bins)
        assert np.array_equal(czbin, cos_zenith_bins)
        e_width, z_width = np.meshgrid(np.ediff1d(enbin), np.ediff1d(czbin))
        return hist_val / (e_width * 2 * np.pi * z_width * nspecies)

    def __add__(self, other):
        return Weighter(self.data + other.data, self.surface + other.surface)

    def tostring(self, flux=None):
        """
        Creates a string with important information about this weighting object:
        generation surface, event map, number of events, and effective area.
        if optional flux is provided the event rate and livetime are added as well.

        """
        output = str(self.surface) + "\n"
        output += pformat(self.colnames) + "\n"
        output += "Number of Events : {:8d}\n".format(len(self.get_weights(1)))
        eff_area = self.effective_area(
            self.surface.get_energy_range(None), self.surface.get_cos_zenith_range(None)
        )
        output += "Effective Area   : {:8.6g} m²\n".format(eff_area[0][0])
        if flux:
            weights = self.get_weights(flux)
            output += "Using flux model : {}\n".format(flux.__class__.__name__)
            output += "Event Rate       : {:8.6g} Hz\n".format(weights.sum())
            output += "Livetime         : {:8.6g} s\n".format(weights.sum() / (weights ** 2).sum())
        return output

    def __str__(self):
        return self.tostring()
