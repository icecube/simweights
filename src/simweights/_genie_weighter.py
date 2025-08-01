# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np

from ._generation_surface import CompositeSurface, GenerationSurface
from ._nugen_weighter import nugen_spatial, nugen_spectrum
from ._pdgcode import PDGCode
from ._powerlaw import PowerLaw
from ._spatial import CircleInjector, SpatialDist
from ._utils import constcol, get_column, get_table, has_column, has_table
from ._weighter import Weighter

if TYPE_CHECKING:  # pragma: no cover
    from np.typing import ArrayLike, NDArray

    from simweights._pdgcode import PDGCode


class GenieSurface(GenerationSurface):
    """Represents a surface on which GENIE simulation was generated on."""

    def __init__(
        self,
        pdgid: PDGCode | int,
        nevents: float,
        global_probability_scale: float,
        power_law: PowerLaw,
        spatial: SpatialDist,
    ) -> None:
        super().__init__(pdgid, nevents, power_law, spatial)
        self.global_probability_scale = global_probability_scale

    def equivalent(self, surface: Any) -> bool:
        """Test for weather two surfaces cand be combined into a single surface with the sum of the nevents."""
        return super().equivalent(surface) and self.global_probability_scale == surface.global_probability_scale

    def get_epdf(self: GenieSurface, weight_cols: Mapping[str, ArrayLike]) -> NDArray[np.float64]:
        """Get the extended pdf of a sample of GENIE."""
        return (
            self.nevents
            / self.global_probability_scale
            / weight_cols["wght"]
            / weight_cols["volscale"]
            / self.spatial.etendue
            * self.power_law.pdf(weight_cols["energy"])
        )

    def __repr__(self) -> str:
        return f"GenieSurface({self.pdgid}, {self.nevents}, {self.global_probability_scale}, {self.power_law}, {self.spatial})"


def genie_icetray_surface(
    mcweightdict: list[Mapping[str, float]], geniedict: Iterable[Mapping[str, float]], nufraction: float = 0.7
) -> CompositeSurface:
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

    surfaces = CompositeSurface()
    for row in unique_schemes:
        (pid, _, _, _, _) = row
        mask = np.all(gen_schemes == row[None, :], axis=1)

        spatial = nugen_spatial(mcweightdict, mask)
        spectrum = nugen_spectrum(mcweightdict, mask)

        type_weight = nufraction if pid > 0 else 1 - nufraction
        n_events = type_weight * constcol(mcweightdict, "NEvents", mask)
        gps = constcol(mcweightdict, "GlobalProbabilityScale", mask)

        surfaces.insert(GenieSurface(pid, n_events, gps, spectrum, spatial))
    return surfaces


def genie_reader_surface(table: Iterable[Mapping[str, float]]) -> CompositeSurface:
    """Inspect the rows of a GENIE S-Frame table object to generate a surface object."""
    surfaces = CompositeSurface()
    n_flux_events = get_column(table, "n_flux_events")
    power_law_index = get_column(table, "power_law_index")
    cylinder_radius = get_column(table, "cylinder_radius")
    max_zenith = get_column(table, "max_zenith")
    min_zenith = get_column(table, "min_zenith")
    min_energy = get_column(table, "min_energy")
    max_energy = get_column(table, "max_energy")
    primary_type = get_column(table, "primary_type")
    n_flux_events = get_column(table, "n_flux_events")
    global_probability_scale = get_column(table, "global_probability_scale")
    for i, nevents in enumerate(n_flux_events):
        assert power_law_index[i] >= 0
        spatial = CircleInjector(
            cylinder_radius[i],
            np.cos(max_zenith[i]),
            np.cos(min_zenith[i]),
        )
        spectrum = PowerLaw(
            -power_law_index[i],
            min_energy[i],
            max_energy[i],
        )
        pdgid = int(primary_type[i])
        surfaces.insert(
            GenieSurface(pdgid, nevents, global_probability_scale[i], spectrum, spatial),
        )
    return surfaces


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

        surface = genie_icetray_surface(weight_table, result_table)
        surface.scale(nfiles)

        momentum_vec = np.array(
            [get_column(result_table, "pxv"), get_column(result_table, "pyv"), get_column(result_table, "pzv")]
        )
        cos_zen = -1 * get_column(result_table, "pzv") / np.sum(momentum_vec**2, axis=0) ** 0.5
        assert cos_zen.min() >= -1
        assert cos_zen.max() <= 1

        weighter = Weighter([file_obj], surface)
        weighter.add_weight_column("pdgid", get_column(result_table, "neu").astype(np.int32))
        weighter.add_weight_column("energy", get_column(result_table, "Ev"))
        weighter.add_weight_column("cos_zen", cos_zen)
        weighter.add_weight_column("wght", get_column(result_table, "wght"))
        weighter.add_weight_column("volscale", np.ones_like(get_column(result_table, "Ev")))

    else:
        msg = (
            "Missing at least one necessary object for GENIE event weighting. If your file is produced by "
            "genie-icetray, be sure to include both the I3MCWeightDict and I3GENIEResultDict in your input "
            "file. If the file is produced by genie-reader, include both I3GenieInfo and I3GenieResult."
        )
        raise KeyError(msg)

    return weighter
