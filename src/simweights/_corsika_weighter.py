# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

import numbers
import warnings
from typing import Any

import numpy as np

from ._generation_surface import GenerationSurface, generation_surface
from ._powerlaw import PowerLaw
from ._spatial import NaturalRateCylinder
from ._utils import constcol, get_column, get_table, has_table
from ._weighter import Weighter


def sframe_corsika_surface(table: Any, oversampling: bool) -> GenerationSurface:
    """Inspect the rows of a CORSIKA S-Frame table object to generate a surface object.

    This function works on files generated with either triggered CORSIKA or corsika-reader because
    `I3PrimaryInjectorInfo` and `I3CorsikaInfo` use exactly the same names for quantities.
    """
    surfaces = []

    for i in range(len(get_column(table, "n_events"))):
        assert get_column(table, "power_law_index")[i] <= 0
        spatial = NaturalRateCylinder(
            get_column(table, "cylinder_height")[i],
            get_column(table, "cylinder_radius")[i],
            np.cos(get_column(table, "max_zenith")[i]),
            np.cos(get_column(table, "min_zenith")[i]),
        )
        spectrum = PowerLaw(
            get_column(table, "power_law_index")[i],
            get_column(table, "min_energy")[i],
            get_column(table, "max_energy")[i],
        )
        oversampling_val = get_column(table, "oversampling")[i] if oversampling else 1
        surfaces.append(
            get_column(table, "n_events")[i]
            * oversampling_val
            * generation_surface(int(get_column(table, "primary_type")[i]), spectrum, spatial),
        )
    retval = sum(surfaces)
    assert isinstance(retval, GenerationSurface)
    return retval


def weight_map_corsika_surface(table: Any) -> GenerationSurface:
    """Inspect the `CorsikaWeightMap` table object of a corsika file to generate a surface object."""
    pdgids = sorted(np.unique(get_column(table, "ParticleType").astype(int)))
    surface: int | GenerationSurface = 0
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
        surface += nevents * generation_surface(pdgid, spectrum, spatial)
    assert isinstance(surface, GenerationSurface)
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

        surface = sframe_corsika_surface(get_table(file_obj, info_obj), oversampling=False)
        triggered = True

    elif nfiles is None:
        info_obj = "I3CorsikaInfo"
        if not has_table(file_obj, info_obj):
            msg = (
                f"File `{getattr(file_obj, 'filename', '<NONE>')}` is was not passed an parameter for "
                "nfiles and no I3CorsikaInfo table was found."
            )
            raise RuntimeError(msg)
        surface = sframe_corsika_surface(get_table(file_obj, info_obj), oversampling=True)
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
        surface = nfiles * weight_map_corsika_surface(table)
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
