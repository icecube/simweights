from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Iterable, Mapping, Optional, Set, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .generation_surface import GenerationSurfaceCollection
from .utils import get_column, get_table


class Weighter:
    """
    Abstract base class from which all weighers derive.

    Weighters will take a file object as input and calculate the weights of the events in the file
    for a given flux. As well as helper functions for all columns in the file. Weighters will keep
    track of the generation surface for the Monte Carlo in question. Weighters can be
    added together to form samples with different simulation parameters
    """

    def __init__(self, data: Iterable[tuple[Any, Mapping]], surface: GenerationSurfaceCollection):
        colnames: Set[str] = set()
        for _, event_map in data:
            keys = set(event_map.keys())
            if colnames:
                colnames = colnames.intersection(keys)
            else:
                colnames = keys

        self.data = list(data)
        self.surface = surface
        self.colnames = sorted(colnames)
        self.__cache: dict = {}

    def get_column(self, table: str, column: str) -> NDArray[np.float64]:
        """
        Helper function to get a specific column from the file
        """
        retval: NDArray = np.array([])
        for datafile, _ in self.data:
            retval = np.append(retval, get_column(get_table(datafile, table), column))
        print("RETVAL", retval)
        return retval

    def get_weight_column(self, name: str) -> NDArray[np.float64]:
        """
        Helper function to get a column needed in the weight calculation
        """
        if name in self.__cache:
            return self.__cache[name]

        if name == "cos_zen":
            retval = np.cos(self.get_weight_column("zenith"))

        else:
            retval = np.array([])
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

        self.__cache[name] = retval
        return retval

    def get_weights(self, flux: Any) -> NDArray[np.float64]:
        """
        Calculate the weights for the sample for the given flux.

        Args:
          flux: can be any one of several types:

            * An instance of :py:class:`nuflux.FluxFunction` from
              `nuflux <https://github.com/icecube/nuflux>`_
            * A callable where the names of the arguments are match the weight objects weighting columns.
            * An iterable the same length os the datasample. If you have other means of calculating
              the flux for each event than the above options this can be useful.
            * A scalar number. This calculates the unrealistic scenario where all events have the same
              flux, this can be useful for testing or calculating effective areas. If the value is 1 then
              the return value will be the well known quantity OneWeight.
        """

        event_col = {k: self.get_weight_column(k) for k in ["energy", "pdgid", "cos_zen"]}

        # do nothing if everything is empty
        if event_col["pdgid"].shape == (0,):
            return np.array([])

        epdf = self.surface.get_epdf(**event_col)
        event_weight = self.get_weight_column("event_weight")

        # calculate the flux based on which type of flux it is
        if hasattr(flux, "getFlux"):
            # this is a nuflux model
            assert callable(flux.getFlux)
            flux_val = flux.getFlux(
                self.get_weight_column("pdgid"),
                self.get_weight_column("energy"),
                self.get_weight_column("cos_zen"),
            )
        elif callable(flux):
            # this is a cosmic ray flux model or just a function
            keys = inspect.signature(flux).parameters.keys()
            try:
                arguments = {k: self.get_weight_column(k) for k in keys}
            except KeyError as missing_params:
                raise ValueError(
                    f"get_weights() was passed callable {repr(flux)} which has parameters {list(keys)}. "
                    "The weight columns which are available are {repr(self.colnames)}"
                ) from missing_params
            flux_val = flux(**arguments)
        elif hasattr(flux, "__len__"):
            # this is an array with a length equal to the number of events
            flux_val = np.array(flux, dtype=float)
        elif np.isscalar(flux):
            # this is a scalar
            flux_val = np.full(epdf.shape, flux)
        else:
            raise ValueError(f"I do not understand what to do with flux {flux}")
        assert flux_val.shape == epdf.shape

        # Getting events with epdf=0 indicates some sort of mismatch between the
        # the surface and the dataset that can't be solved here so print a
        # warning and ignore the events
        mask = epdf > 0

        if not np.all(mask):
            warnings.warn(
                f"simweights :: {np.logical_not(mask).sum()} events out of {mask.size} were found to be "
                "outside the generation surface. This could indicate a problem with this dataset."
            )
        weights = np.zeros_like(epdf)
        weights[mask] = (event_weight * flux_val)[mask] / epdf[mask]
        return weights

    def effective_area(
        self, energy_bins: ArrayLike, cos_zenith_bins: ArrayLike, mask: Optional[ArrayLike] = None
    ) -> NDArray[np.float64]:
        r"""
        Calculate The effective area for the given energy and zenith bins.

        This is accomplished by histogramming the generation surface the simulation sample
        in energy and zenith bins and dividing by the size of the energy and solid angle of each bin.
        If mask is passed as a parameter, only events which are included in the mask are used.
        Effective areas are given units of :math:`\mathrm{m}^2`

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

        weights = self.get_weights(1e-4)
        if mask is None:
            maska = np.full(weights.size, 1, dtype=bool)
        else:
            maska = np.asarray(mask, dtype=bool)

        assert maska.shape == weights.shape

        hist_val, czbin, enbin = np.histogram2d(
            cos_zen[maska],
            energy[maska],
            weights=weights[maska],
            bins=[cos_zenith_bins, energy_bins],
        )

        nspecies = len(np.unique(self.get_weight_column("pdgid")[maska]))

        assert np.array_equal(enbin, energy_bins)
        assert np.array_equal(czbin, cos_zenith_bins)
        e_width, z_width = np.meshgrid(np.ediff1d(enbin), np.ediff1d(czbin))
        return hist_val / (e_width * 2 * np.pi * z_width * nspecies)

    def __add__(self, other: Weighter) -> Weighter:
        return Weighter(self.data + other.data, self.surface + other.surface)

    def tostring(self, flux: Union[None, object, Callable, ArrayLike] = None) -> str:
        """
        Creates a string with important information about this weighting object:
        generation surface, event map, number of events, and effective area.
        if optional flux is provided the event rate and livetime are added as well.

        """
        output = str(self.surface) + "\n"
        output += f"Weight Columns   : {', '.join(self.colnames)}\n"
        output += f"Number of Events : {len(self.get_weights(1)):8d}\n"
        eff_area = self.effective_area(
            self.surface.get_energy_range(None), self.surface.get_cos_zenith_range(None)
        )
        output += f"Effective Area   : {eff_area[0][0]:8.6g} m²\n"
        if flux:
            weights = self.get_weights(flux)
            output += f"Using flux model : {flux.__class__.__name__}\n"
            output += f"Event Rate       : {weights.sum():8.6g} Hz\n"
            output += f"Livetime         : {weights.sum() / (weights ** 2).sum():8.6g} s\n"
        return output

    def __str__(self) -> str:
        return self.tostring()
