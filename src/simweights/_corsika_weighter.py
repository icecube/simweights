# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

import numbers
import warnings
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from ._generation_surface import CompositeSurface, GenerationSurface
from ._powerlaw import PowerLaw
from ._spatial import NaturalRateCylinder
from ._utils import constcol, get_column, get_table, has_column, has_table
from ._weighter import Weighter

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CorsikaSurface(GenerationSurface):
    """Represents a surface on which CORSIKA simulation was generated on."""

    def get_epdf(self: CorsikaSurface, weight_cols: Mapping[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Get the extended pdf of a sample of CORSIKA."""
        return (
            self.nevents
            / weight_cols["event_weight"]
            * self.power_law.pdf(weight_cols["energy"])
            * self.spatial.pdf(weight_cols["cos_zen"])
        )


def sframe_corsika_surface(table: Any) -> CompositeSurface:
    """Inspect the rows of a CORSIKA S-Frame table object to generate a surface object.

    This function works on files generated with either triggered CORSIKA or corsika-reader because
    `I3PrimaryInjectorInfo` and `I3CorsikaInfo` use exactly the same names for quantities.
    """
    surfaces = CompositeSurface()
    cylinder_height = get_column(table, "cylinder_height")
    cylinder_radius = get_column(table, "cylinder_radius")
    max_zenith = get_column(table, "max_zenith")
    min_zenith = get_column(table, "min_zenith")
    power_law_index = get_column(table, "power_law_index")
    min_energy = get_column(table, "min_energy")
    max_energy = get_column(table, "max_energy")
    n_events = get_column(table, "n_events")
    for i in range(len(get_column(table, "n_events"))):
        assert get_column(table, "power_law_index")[i] <= 0
        spatial = NaturalRateCylinder(
            cylinder_height[i],
            cylinder_radius[i],
            np.cos(max_zenith[i]),
            np.cos(min_zenith[i]),
        )
        spectrum = PowerLaw(
            power_law_index[i],
            min_energy[i],
            max_energy[i],
        )
        oversampling_val = get_column(table, "oversampling")[i] if has_column(table, "oversampling") else 1
        pdgid = int(get_column(table, "primary_type")[i])
        surfaces.insert(CorsikaSurface(pdgid, n_events[i] * oversampling_val, spectrum, spatial))
    return surfaces


def weight_map_corsika_surface(table: Any) -> CompositeSurface:
    """Inspect the `CorsikaWeightMap` table object of a corsika file to generate a surface object."""
    pdgids = sorted(np.unique(get_column(table, "ParticleType").astype(int)))

    if len(pdgids) == 0:
        msg = "`CorsikaWeightMap` is empty. SimWeights cannot process this file"
        raise RuntimeError(msg)
    surface = CompositeSurface()
    for pdgid in pdgids:
        mask = pdgid == get_column(table, "ParticleType")

        spatial = NaturalRateCylinder(
            constcol(table, "CylinderLength", mask),
            constcol(table, "CylinderRadius", mask),
            np.cos(constcol(table, "ThetaMax", mask)),
            np.cos(constcol(table, "ThetaMin", mask)),
        )

        primary_spectral_index = round(constcol(table, "PrimarySpectralIndex", mask), 6)
        assert primary_spectral_index <= 0

        spectrum = PowerLaw(
            primary_spectral_index,
            constcol(table, "EnergyPrimaryMin", mask),
            constcol(table, "EnergyPrimaryMax", mask),
        )
        nevents = constcol(table, "OverSampling", mask) * constcol(table, "NEvents", mask)
        surface.insert(CorsikaSurface(pdgid, nevents, spectrum, spatial))
    return surface


def CorsikaWeighter(file_obj: Any, nfiles: float | None = None) -> Weighter:  # noqa: N802
    # pylint: disable=invalid-name
    """Weighter for CORSIKA-in-ice simulation made with I3CORSIKAReader.

    I3CORSIKAReader not use S-Frames and stores the surface information in an I3MapStringDouble so that
    the user does not know how many jobs contributed to the current sample, so it needs to know the number
    of files.
    """
    if has_table(file_obj, "I3CorsikaWeight"):
        if nfiles is not None:
            mesg = (
                f"This file, `{getattr(file_obj, 'filename', '<NONE>')}`, was identified as a triggered "
                "CORSIKA file based on the existence of the `I3CorsikaWeight` object. However the "
                "parameter nfiles was passed to CorsikaWeighter. This is unnecessary."
            )
            raise RuntimeError(mesg)

        info_obj = "I3PrimaryInjectorInfo"
        if not has_table(file_obj, info_obj):
            mesg = (
                f"This file, `{getattr(file_obj, 'filename', '<NONE>')}`, was identified as a triggered "
                "CORSIKA file based on the existence of the `I3CorsikaWeight` object. However it seems to "
                "be missing the S-Frames table `I3PrimaryInjectorInfo`."
            )
            raise RuntimeError(mesg)

        surface = sframe_corsika_surface(get_table(file_obj, info_obj))
        triggered = True

    elif nfiles is None:
        info_obj = "I3CorsikaInfo"
        if not has_table(file_obj, info_obj):
            msg = (
                f"File `{getattr(file_obj, 'filename', '<NONE>')}` is was not passed an parameter for "
                "nfiles and no I3CorsikaInfo table was found."
            )
            raise RuntimeError(msg)
        surface = sframe_corsika_surface(get_table(file_obj, info_obj))
        triggered = False

    else:
        if not isinstance(nfiles, numbers.Number):
            raise TypeError(
                "CorsikaWeighter: argument nfiles must be a floating point number. Got " + str(nfiles),
            )

        if has_table(file_obj, "I3CorsikaInfo"):
            warnings.warn(
                "CorsikaWeighter was given a value for nfiles, but this file has an "
                "I3CorsikaInfo table indicating it has S-Frames",
                stacklevel=2,
            )

        table = get_table(file_obj, "CorsikaWeightMap")
        surface = weight_map_corsika_surface(table)
        surface.scale(nfiles)
        triggered = False

    weighter = Weighter([file_obj], surface)

    if triggered:
        weighter.add_weight_column("pdgid", weighter.get_column("I3CorsikaWeight", "type"))
        weighter.add_weight_column("energy", weighter.get_column("I3CorsikaWeight", "energy"))
        weighter.add_weight_column("cos_zen", np.cos(weighter.get_column("I3CorsikaWeight", "zenith")))
        weighter.add_weight_column("event_weight", weighter.get_column("I3CorsikaWeight", "weight"))
    else:
        energy = weighter.get_column("PolyplopiaPrimary", "energy")
        weighter.add_weight_column("energy", energy)
        weighter.add_weight_column("pdgid", weighter.get_column("PolyplopiaPrimary", "type"))
        weighter.add_weight_column("cos_zen", np.cos(weighter.get_column("PolyplopiaPrimary", "zenith")))
        weighter.add_weight_column("event_weight", np.full(len(energy), 1))

    return weighter
