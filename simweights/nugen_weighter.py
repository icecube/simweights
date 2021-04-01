from copy import copy

import numpy as np

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

    @staticmethod
    def _get_surface_map(infile):
        # nugen generates an equal number of nu's and nubar's so you need to make two surfaces
        # each with an n_events of half the number in the NEvents column
        table = get_table(infile, "I3MCWeightDict")
        neutrino_surface = dict(
            n_events=0.5 * get_constant_column(get_column(table, "NEvents")),
            primary_type=get_constant_column(abs(get_column(table, "PrimaryNeutrinoType"))),
            cylinder_height=get_constant_column(get_column(table, "CylinderHeight")),
            cylinder_radius=get_constant_column(get_column(table, "CylinderRadius")),
            min_energy=10 ** get_constant_column(get_column(table, "MinEnergyLog")),
            max_energy=10 ** get_constant_column(get_column(table, "MaxEnergyLog")),
            min_zenith=get_constant_column(get_column(table, "MinZenith")),
            max_zenith=get_constant_column(get_column(table, "MaxZenith")),
            power_law_index=-get_constant_column(get_column(table, "PowerLawIndex")),
        )
        # create a copy for the nubar and change the value of the type
        anti_neutrino_surface = copy(neutrino_surface)
        anti_neutrino_surface.update(dict(primary_type=-neutrino_surface["primary_type"]))
        assert neutrino_surface["primary_type"] in [12, 14, 16]
        assert anti_neutrino_surface["primary_type"] in [-12, -14, -16]
        return [neutrino_surface, anti_neutrino_surface]

    def _get_surface_params(self):
        return dict(
            energy=self.get_column("I3MCWeightDict", "PrimaryNeutrinoEnergy"),
            particle_type=self.get_column("I3MCWeightDict", "PrimaryNeutrinoType"),
            cos_zen=np.cos(self.get_column("I3MCWeightDict", "PrimaryNeutrinoZenith")),
        )

    def _get_flux_params(self):
        return self._get_surface_params()

    def _get_event_weight(self):
        return self.get_column("I3MCWeightDict", "TotalWeight")
