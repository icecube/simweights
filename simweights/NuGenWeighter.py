from copy import copy

import numpy as np

from .utils import get_column, get_constant_column, get_table
from .WeighterBase import MapWeighter


class NuGenWeighter(MapWeighter):
    @staticmethod
    def _get_surface_map(infile):
        table = get_table(infile, "I3MCWeightDict")
        d1 = dict(
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
        d2 = copy(d1)
        d2.update(dict(primary_type=-d1["primary_type"]))
        return [d1, d2]

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
