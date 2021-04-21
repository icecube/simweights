import numpy as np

from .cylinder import UniformSolidAngleCylinder
from .utils import get_column, get_constant_column, get_table
from .weighter import MapWeighter


class NuGenWeighter(MapWeighter):
    """
    Weighter for neutrino-generator (NuGen) simulation

    Does not use S-Frames and stores the surface information in an I3MapStringDouble so that the user
    does not know how many jobs contributed to the current sample, so it needs to know the number of
    files. Nugen calculates the event weight in a column called ``TotalWeight`` which takes into account
    the netutrino cross-section, detector density, and distance traveled through the generation volume.
    """

    _spatial_distribution = UniformSolidAngleCylinder

    @staticmethod
    def _get_surface_map(infile):
        # nugen generates both nu and nubar with the fraction stored in TypeWeight and the total stored
        # in NEvents
        table = get_table(infile, "I3MCWeightDict")
        pdgids = np.unique(get_column(table, "PrimaryNeutrinoType"))
        surfaces = []
        for pid in pdgids:
            mask = pid == get_column(table, "PrimaryNeutrinoType")
            surfaces.append(
                dict(
                    n_events=get_constant_column(get_column(table, "TypeWeight")[mask])
                    * get_constant_column(get_column(table, "NEvents")[mask]),
                    primary_type=get_constant_column(get_column(table, "PrimaryNeutrinoType")[mask]),
                    cylinder_height=get_constant_column(get_column(table, "CylinderHeight")[mask]),
                    cylinder_radius=get_constant_column(get_column(table, "CylinderRadius")[mask]),
                    min_energy=10 ** get_constant_column(get_column(table, "MinEnergyLog")[mask]),
                    max_energy=10 ** get_constant_column(get_column(table, "MaxEnergyLog")[mask]),
                    min_zenith=get_constant_column(get_column(table, "MinZenith")[mask]),
                    max_zenith=get_constant_column(get_column(table, "MaxZenith")[mask]),
                    power_law_index=-get_constant_column(get_column(table, "PowerLawIndex")[mask]),
                )
            )
        return surfaces

    def _get_surface_params(self):
        return dict(
            energy=self.get_column("I3MCWeightDict", "PrimaryNeutrinoEnergy"),
            pdgid=self.get_column("I3MCWeightDict", "PrimaryNeutrinoType"),
            cos_zen=np.cos(self.get_column("I3MCWeightDict", "PrimaryNeutrinoZenith")),
        )

    def _get_flux_params(self):
        return self._get_surface_params()

    def _get_event_weight(self):
        return self.get_column("I3MCWeightDict", "TotalWeight")
