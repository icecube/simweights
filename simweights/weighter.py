import warnings
from copy import copy

import numpy as np

from .generation_surface import GenerationSurface
from .powerlaw import PowerLaw
from .utils import Null, get_column, get_table


class Weighter:
    """
    Abstract base class from which all weighers derive.

    Weighters will take a file object as input and calculate the weights of the events in the file
    for a given flux. As well as helper functions for all columns in the file. Weighters will keep
    track of the generation surface for the Monte Carlo in question. Weighters can be
    added together to form samples with differnt simulation parameters
    """

    _spatial_distribution = NotImplementedError

    def __init__(self, surface, data):
        self.surface = surface
        self.data = data

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
        epdf = self.surface.get_epdf(**self._get_surface_params())
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

        sparam = self._get_surface_params()
        if pdgid is None:
            mask = np.ones_like(sparam["pdgid"], dtype=bool)
            nspecies = len(self.surface.get_pdgids())
        else:
            mask = pdgid == sparam["pdgid"]
            nspecies = 1

        weights = self.get_weights(lambda energy, pdgid: 1)
        hist_val, czbin, enbin = np.histogram2d(
            sparam["cos_zen"][mask],
            sparam["energy"][mask],
            weights=weights[mask],
            bins=[cos_zenith_bins, energy_bins],
        )

        assert np.array_equal(enbin, energy_bins)
        assert np.array_equal(czbin, cos_zenith_bins)
        e_width, z_width = np.meshgrid(np.ediff1d(enbin), np.ediff1d(czbin))
        return hist_val / (e_width * 2 * np.pi * z_width * nspecies)

    @classmethod
    def _get_surface(cls, smap):
        assert smap["power_law_index"] <= 0
        spatial = cls._spatial_distribution(
            smap["cylinder_height"],
            smap["cylinder_radius"],
            np.cos(smap["max_zenith"]),
            np.cos(smap["min_zenith"]),
        )
        spectrum = PowerLaw(smap["power_law_index"], smap["min_energy"], smap["max_energy"])
        return GenerationSurface(smap["primary_type"], smap["n_events"], spectrum, spatial)

    def _get_surface_params(self):
        raise NotImplementedError()

    def _get_flux_params(self):
        raise NotImplementedError()

    def _get_event_weight(self):
        raise NotImplementedError()

    def __add__(self, other):
        if type(self) is not type(other):
            raise ValueError("Cannot add {} to {}".format(type(self), type(self)))
        ret = copy(self)
        ret.surface += other.surface
        ret.data = self.data + other.data
        return ret


class MapWeighter(Weighter):
    """
    Abstract base class for weighter which don't have S-frames

    These generators (CorsikaReader and neutrion-generator) store the surface information in an
    I3MapStringDouble and do not know how many jobs contributed to the current sample.
    So the user must provide nfiles
    """

    def __init__(self, infile, nfiles):
        assert nfiles is not None
        surface = Null()
        for smap in self._get_surface_map(infile):
            surface += nfiles * self._get_surface(smap)
        super().__init__(surface, [infile])

    @staticmethod
    def _get_surface_map(infile):
        raise NotImplementedError()
