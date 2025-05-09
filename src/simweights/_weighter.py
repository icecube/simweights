# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np
from scipy.integrate import quad  # pylint: disable=import-error

from ._utils import get_column, get_table

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ._generation_surface import GenerationSurface  # pragma: nocover


class Weighter:
    """Abstract base class from which all weighers derive.

    Weighters will take a file object as input and calculate the weights of the events in the file
    for a given flux. As well as helper functions for all columns in the file. Weighters will keep
    track of the generation surface for the Monte Carlo in question. Weighters can be
    added together to form samples with different simulation parameters
    """

    def __init__(self: Weighter, data: Iterable[Any], surface: GenerationSurface) -> None:
        self.data = list(data)
        self.surface = surface
        self.weight_cols: dict[str, NDArray[np.float64]] = {}
        self.colnames = sorted(self.weight_cols.keys())
        self.size: int | None = None

    def get_column(self: Weighter, table: str, column: str) -> NDArray[np.float64]:
        """Helper function to get a specific column from the file."""
        retval: NDArray[np.float64] = np.array([])
        for datafile in self.data:
            retval = np.append(retval, get_column(get_table(datafile, table), column))
        return retval

    def add_weight_column(self: Weighter, name: str, column: ArrayLike) -> None:
        """Add a new column to be passed as parameters to flux models."""
        col = np.array(column)
        if self.size:
            if self.size != col.size:
                mesg = f"{self.size}  != {col.size}"
                raise ValueError(mesg)
        else:
            self.size = col.size

        self.weight_cols[name] = col
        self.colnames = sorted(self.weight_cols.keys())

    def get_weight_column(self: Weighter, name: str) -> NDArray[np.float64]:
        """Helper function to get a column needed in the weight calculation."""
        return self.weight_cols[name]

    def get_weights(self: Weighter, flux: Any) -> NDArray[np.float64]:
        """Calculate the weights for the sample for the given flux.

        Args:
            flux (Callable | FluxFunction | ArrayLike ):
                A model describing the flux of the primaries to weight against

        The Flux argument can be one of several types:

        *   An instance of :py:class:`nuflux.FluxFunction` from
            `nuflux <https://github.com/icecube/nuflux>`_.
        *   A callable where the names of the arguments match the weight objects weighting
            columns.
        *   An iterable with the same length as the data sample. If you have other means
            of calculating the flux for each event than the above options this can be
            useful.
        *   A scalar number, this calculates the unrealistic scenario where all events
            have the same flux, this can be useful for testing or calculating effective
            areas. For neutrinos, If the value is 1 then the return value will be the
            well known quantity OneWeight.
        """
        # get a dictionary of the columns we need for weighting
        event_col = {k: self.get_weight_column(k) for k in ["pdgid", *self.surface.get_keys()]}

        # do nothing if everything is empty
        if event_col["pdgid"].shape == (0,):
            return np.array([])

        epdf = self.surface.get_epdf(**event_col)

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
                mesg = (
                    f"get_weights() was passed callable {flux!r} which has parameters {list(keys)}. "
                    f"The weight columns which are available are {self.colnames!r}"
                )
                raise ValueError(mesg) from missing_params
            flux_val = flux(**arguments)
        elif hasattr(flux, "__len__"):
            # this is an array with a length equal to the number of events
            flux_val = np.array(flux, dtype=float)
        elif np.isscalar(flux):
            # this is a scalar
            flux_val = np.full(epdf.shape, flux)
        else:
            mesg = f"I do not understand what to do with flux {flux}"
            raise ValueError(mesg)
        assert flux_val.shape == epdf.shape

        # Getting events with epdf=0 indicates some sort of mismatch between the
        # the surface and the dataset that can't be solved here so print a
        # warning and ignore the events
        mask = epdf > 0

        if not np.all(mask):
            warnings.warn(
                f"simweights :: {np.logical_not(mask).sum()} events out of {mask.size} were found to be "
                "outside the generation surface. This could indicate a problem with this dataset.",
                stacklevel=2,
            )
        weights = np.zeros_like(epdf)
        weights[mask] = flux_val[mask] / epdf[mask]
        return weights

    def effective_area(
        self: Weighter,
        energy_bins: ArrayLike,
        cos_zenith_bins: ArrayLike,
        mask: ArrayLike | None = None,
        flux: Any = 1,  # default is 1 GeV^-1 cm^-2 sr^-1 flux
        *,
        return_stddev: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Calculate the effective area for the given energy and zenith bins.

        This is accomplished by histogramming the generation surface of the simulation sample
        in energy and zenith bins and dividing by the size of the energy and solid angle of each bin.
        If a mask is passed as a parameter, only events which are included in the mask are used.
        Effective areas are given in units of :math:`\mathrm{m}^2`

        .. Note ::

            If the sample contains more than one type of primary particle, then the result will be
            averaged over the number of particles. This is usually what you want. However, it can
            cause strange behavior if there is a small number of one type. In this case, the mask
            should be used to select the particle types individually. Alternatively, a flux model
            can be specified to apply a weighting to the individual primary particle types. If
            return_stddev is set to True, a tuple of effective area histograms and their standard
            deviation are returned.


        Args:
            energy_bins: array_like
                A length N+1 array of energy bin edges
            cos_zenith_bins: array_like
                A length M+1 array of cos(zenith) bin edges
            mask: array_like
                boolean array where 1 indicates to use the event in the calculation.
                Must have the same length as the simulation sample.
            flux: Any
                A model describing the flux of the primaries to weight against. For
                possible types, see get_weights()
            return_stddev: bool
                boolean to specify if only effective area (default) or a tuple of
                effective area and its standard deviation is returned

        Returns:
            array_like | tuple(array_like, array_like)
                An NxM array of effective areas. Where N is the number of energy bins and M
                is the number of cos(zenith) bins.
                If return_stddev, a tuple of two NxM arrays (effective area and its
                standard deviation)

        """
        cm2_to_m2 = 1e-4
        energy_bins = np.array(energy_bins)
        cos_zenith_bins = np.array(cos_zenith_bins)

        assert energy_bins.ndim == 1
        assert cos_zenith_bins.ndim == 1
        assert len(energy_bins) >= 2  # noqa: PLR2004
        assert len(cos_zenith_bins) >= 2  # noqa: PLR2004

        energy = self.get_weight_column("energy")
        cos_zen = self.get_weight_column("cos_zen")

        weights = self.get_weights(flux)
        maska = np.full(weights.size, 1, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

        assert maska.shape == weights.shape

        hist_val, czbin, enbin = np.histogram2d(
            cos_zen[maska],
            energy[maska],
            weights=weights[maska],
            bins=[cos_zenith_bins, energy_bins],
        )

        hist_val_variance, _, _ = np.histogram2d(
            cos_zen[maska],
            energy[maska],
            weights=(weights[maska]) ** 2,
            bins=[cos_zenith_bins, energy_bins],
        )

        assert np.array_equal(enbin, energy_bins)
        assert np.array_equal(czbin, cos_zenith_bins)
        pdgids = np.unique(self.get_weight_column("pdgid")[maska])

        if np.isscalar(flux):
            flux_integrals = len(pdgids) * np.ediff1d(enbin)
        elif callable(flux):
            flux_integrals = np.asarray(
                [
                    sum(quad(flux, energy_bins[bin_index], energy_bins[bin_index + 1], args=(p,))[0] for p in pdgids)
                    for bin_index in range(len(energy_bins) - 1)
                ]
            )
        else:
            mesg = f"flux of type {type(flux)} is supplied but only scalar flux or cosmic ray flux models are supported so far"
            raise TypeError(mesg)
        e_int, z_int = np.meshgrid(flux_integrals, np.ediff1d(czbin))
        output: NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]
        if return_stddev:
            output = (
                np.asarray(cm2_to_m2 * hist_val / (e_int * 2 * np.pi * z_int), dtype=np.float64),
                np.asarray(cm2_to_m2 * np.sqrt(hist_val_variance) / (e_int * 2 * np.pi * z_int), dtype=np.float64),
            )
        else:
            output = np.asarray(cm2_to_m2 * hist_val / (e_int * 2 * np.pi * z_int), dtype=np.float64)
        return output

    def __add__(self: Weighter, other: Weighter | int) -> Weighter:
        if other == 0:
            return self
        if not isinstance(other, Weighter):
            mesg = f"cannot add {other!r} to weighter object {self!r}"
            raise TypeError(mesg)

        weighter = Weighter(self.data + other.data, self.surface + other.surface)

        for colname, column in self.weight_cols.items():
            if colname in other.weight_cols:
                weighter.add_weight_column(colname, np.append(column, other.weight_cols[colname]))
        return weighter

    def __radd__(self: Weighter, other: Weighter | int) -> Weighter:
        return self + other

    def tostring(self: Weighter, flux: None | object | Callable[[Any], ArrayLike] | ArrayLike = None) -> str:
        """Creates a string with important information about this weighting object.

        Generation surface, event map, number of events, and effective area.
        If optional flux is provided the event rate and livetime are added as well.

        """
        output = str(self.surface) + "\n"
        output += f"Weight Columns   : {', '.join(self.colnames)}\n"
        output += f"Number of Events : {len(self.get_weights(1)):8d}\n"
        eff_area = self.effective_area(
            self.surface.get_energy_range(None),
            self.surface.get_cos_zenith_range(None),
        )
        output += f"Effective Area   : {eff_area[0][0]:8.6g} m²\n"
        if flux:
            weights = self.get_weights(flux)
            output += f"Using flux model : {flux.__class__.__name__}\n"
            output += f"Event Rate       : {weights.sum():8.6g} Hz\n"
            output += f"Livetime         : {weights.sum() / (weights**2).sum():8.6g} s\n"
        return output

    def __str__(self: Weighter) -> str:
        return self.tostring()
