# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from ._generation_surface import GenerationSurface, generation_surface
from ._nugen_weighter import nugen_spatial, nugen_spectrum
from ._powerlaw import PowerLaw
from ._spatial import CircleInjector
from ._utils import Column, Const, constcol, get_column, get_table, has_column, has_table
from ._weighter import Weighter


def genie_icetray_surface(
    mcweightdict: list[Mapping[str, float]], geniedict: Iterable[Mapping[str, float]], nufraction: float = 0.7
) -> GenerationSurface:
    """Inspect the rows of a GENIE-icetray"s I3MCWeightDict table object to generate a surface object.

    This is a bit of a pain: the oscillations group historically produced 4-5 energy bands with varying
    generation parameters, then merged them all into one "file". Because of this, we need to check the
    neutrino type, volume, and spectrum to get the correct surfaces. The type weight also isn"t stored
    in the files: this was fixed to 70/30 for oscillation-produced genie-icetray files.
    """
    gen_schemes = np.array(
        [
            get_column(geniedict, "neu"),
            get_column(mcweightdict, "GeneratorVolume"),
            get_column(mcweightdict, "PowerLawIndex"),
            get_column(mcweightdict, "MinEnergyLog"),
            get_column(mcweightdict, "MaxEnergyLog"),
        ]
    ).T
    unique_schemes = np.unique(gen_schemes, axis=0)

    if len(unique_schemes) == 0:
        msg = "`I3MCWeightDict` is empty. SimWeights cannot process this file"
        raise RuntimeError(msg)

    surfaces = []
    for row in unique_schemes:
        (pid, _, _, _, _) = row
        mask = np.all(gen_schemes == row[None, :], axis=1)

        spatial = nugen_spatial(mcweightdict[mask])
        spectrum = nugen_spectrum(mcweightdict[mask])

        type_weight = nufraction if pid > 0 else 1 - nufraction
        n_events = type_weight * constcol(mcweightdict, "NEvents", mask)

        surfaces.append(n_events * generation_surface(pid, Column("wght"), spectrum, spatial))
    ret = sum(surfaces)
    assert isinstance(ret, GenerationSurface)
    return ret


def genie_reader_surface(table: Iterable[Mapping[str, float]]) -> GenerationSurface:
    """Inspect the rows of a GENIE S-Frame table object to generate a surface object."""
    surfaces = []

    for i in range(len(get_column(table, "n_flux_events"))):
        assert get_column(table, "power_law_index")[i] >= 0
        spatial = CircleInjector(
            get_column(table, "cylinder_radius")[i],
            np.cos(get_column(table, "max_zenith")[i]),
            np.cos(get_column(table, "min_zenith")[i]),
        )
        spectrum = PowerLaw(
            -get_column(table, "power_law_index")[i],
            get_column(table, "min_energy")[i],
            get_column(table, "max_energy")[i],
            "energy",
        )
        pdgid = int(get_column(table, "primary_type")[i])
        nevents = get_column(table, "n_flux_events")[i]
        global_probability_scale = get_column(table, "global_probability_scale")[i]

        const_factor = 1 / spatial.etendue / global_probability_scale
        surfaces.append(
            nevents * generation_surface(pdgid, Const(const_factor), Column("wght"), Column("volscale"), spectrum),
        )
    retval = sum(surfaces)
    assert isinstance(retval, GenerationSurface)
    return retval


def GenieWeighter(file_obj: Any, nfiles: float | None = None) -> Weighter:  # noqa: N802
    # pylint: disable=invalid-name
    """Weighter for GENIE simulation.

    Reads ``I3GenieInfo`` from S-Frames and ``I3GenieResult`` from Q-Frames for genie-reader files
    and ``I3MCWeightDict`` and "I3GENIEResultDict" from Q-Frames for older legacy genie-icetray files.
    """
    if not any(has_table(file_obj, colname) for colname in ["I3GenieInfo", "I3GenieResult", "I3GENIEResultDict"]):
        msg = (
            f"The file `{getattr(file_obj, 'filename', '<NONE>')}` does not contain at least one of I3GenieInfo, "
            "I3GenieResult, or I3GENIEResultDict, so this is unlikely to be a GENIE file."
        )
        raise TypeError(msg)
    if has_table(file_obj, "I3GenieInfo") and has_table(file_obj, "I3GenieResult"):
        # Branch for newer genie-reader files
        if nfiles is not None:
            msg = (
                f"GenieWeighter received an nfiles={nfiles}, but `{getattr(file_obj, 'filename', '<NONE>')}` "
                "was produced with genie-reader instead of genie-icetray. We expect to read the number of "
                "files from the number of observed S-frames in the file, so this is unnecessary. Do not pass "
                "in a value for nfiles for genie-reader files."
            )
            raise RuntimeError(msg)

        info_table = get_table(file_obj, "I3GenieInfo")
        result_table = get_table(file_obj, "I3GenieResult")
        surface = genie_reader_surface(info_table)

        weighter = Weighter([file_obj], surface)
        weighter.add_weight_column("pdgid", get_column(result_table, "neu").astype(np.int32))
        weighter.add_weight_column("energy", get_column(result_table, "Ev"))
        weighter.add_weight_column("cos_zen", get_column(result_table, "pzv"))
        weighter.add_weight_column("wght", get_column(result_table, "wght"))

        # Include the effect of the muon scaling introduced in icecube/icetray#3607, if present.
        if has_column(result_table, "volscale"):
            volscale = get_column(result_table, "volscale")
        else:
            volscale = np.ones_like(get_column(result_table, "wght"))
        weighter.add_weight_column("volscale", volscale)

    elif has_table(file_obj, "I3MCWeightDict") and has_table(file_obj, "I3GENIEResultDict"):
        # Branch for older genie-icetray files
        if nfiles is None:
            msg = (
                f"GenieWeighter received an nfiles={nfiles}, but `{getattr(file_obj, 'filename', '<NONE>')}` "
                "was produced with genie-icetray instead of genie-reader. We require the number of files to be "
                "passed in for genie-icetray files since we can't simply count S-frames."
            )
            raise RuntimeError(msg)

        weight_table = get_table(file_obj, "I3MCWeightDict")
        result_table = get_table(file_obj, "I3GENIEResultDict")

        surface = nfiles * genie_icetray_surface(weight_table, result_table)

        momentum_vec = np.array(
            [get_column(result_table, "pxv"), get_column(result_table, "pyv"), get_column(result_table, "pzv")]
        )
        cos_zen = -1 * get_column(result_table, "pzv") / np.sum(momentum_vec**2, axis=0) ** 0.5

        weighter = Weighter([file_obj], surface)
        weighter.add_weight_column("pdgid", get_column(result_table, "neu").astype(np.int32))
        weighter.add_weight_column("energy", get_column(result_table, "Ev"))
        weighter.add_weight_column("cos_zen", cos_zen)
        weighter.add_weight_column("wght", get_column(result_table, "wght") * get_column(result_table, "_glbprbscale"))

    else:
        msg = (
            "Missing at least one necessary object for GENIE event weighting. If your file is produced by "
            "genie-icetray, be sure to include both the I3MCWeightDict and I3GENIEResultDict in your input "
            "file. If the file is produced by genie-reader, include both I3GenieInfo and I3GenieResult."
        )
        raise KeyError(msg)

    return weighter
