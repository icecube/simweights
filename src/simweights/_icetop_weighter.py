# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause


from typing import Any, Mapping

import numpy as np

# if TYPE_CHECKING:
from numpy.typing import ArrayLike, NDArray

from ._generation_surface import CompositeSurface, GenerationSurface
from ._powerlaw import PowerLaw
from ._spatial import NaturalRateCylinder
from ._utils import get_column, get_table
from ._weighter import Weighter


class IceTopSurface(GenerationSurface):
    """Represents a surface on which IceTop simulation was generated on."""

    def get_epdf(self: "IceTopSurface", weight_cols: Mapping[str, ArrayLike]) -> "NDArray[np.float64]":
        """Get the extended pdf of a sample of GENIE."""
        return self.nevents * self.power_law.pdf(weight_cols["energy"]) * self.spatial.pdf(weight_cols["cos_zen"])


def sframe_icetop_surface(table: Any) -> CompositeSurface:
    """Inspect the rows of a I3TopInjectorInfo S-Frame table object to generate a surface object."""
    surfaces = CompositeSurface()

    n_events = get_column(table, "n_events")
    power_law_index = get_column(table, "power_law_index")
    min_energy = get_column(table, "min_energy")
    max_energy = get_column(table, "max_energy")
    sampling_radius = get_column(table, "sampling_radius")
    max_zenith = np.cos(get_column(table, "max_zenith"))
    min_zenith = np.cos(get_column(table, "min_zenith"))
    primary_type = get_column(table, "primary_type")

    for i, nevents in enumerate(n_events):
        assert power_law_index[i] <= 0
        spectrum = PowerLaw(  # pylint: disable=duplicate-code
            power_law_index[i],
            min_energy[i],
            max_energy[i],
        )
        spatial = NaturalRateCylinder(
            0,  # set cylinder height to 0 to get simple surface plane
            sampling_radius[i],
            max_zenith[i],
            min_zenith[i],
        )
        pdgid = int(primary_type[i])
        surfaces.insert(IceTopSurface(pdgid, nevents, spectrum, spatial))
    assert isinstance(surfaces, CompositeSurface)
    return surfaces


def IceTopWeighter(file_obj: Any) -> Weighter:  # noqa: N802
    # pylint: disable=invalid-name
    """Weighter for IceTop CORSIKA simulation made with I3TopSimulator.cxx."""
    surface = sframe_icetop_surface(get_table(file_obj, "I3TopInjectorInfo"))
    weighter = Weighter([file_obj], surface)

    pdgid = weighter.get_column("MCPrimary", "type")
    weighter.add_weight_column("pdgid", pdgid)
    weighter.add_weight_column("energy", weighter.get_column("MCPrimary", "energy"))
    weighter.add_weight_column("cos_zen", np.cos(weighter.get_column("MCPrimary", "zenith")))
    return weighter
