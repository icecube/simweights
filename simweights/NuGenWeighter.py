import numpy as np

from .cylinder import VolumeCorrCylinder
from .GenerationSurface import GenerationSurface, GenerationSurfaceCollection
from .powerlaw import PowerLaw
from .utils import Null, get_column, get_constant_column, get_table
from .WeighterBase import Weighter


class NuGenWeighter(Weighter):
    def __init__(self, infile, nfiles):
        assert nfiles is not None
        surface = Null()
        for i in self.read_weight_map(get_table(infile, "I3MCWeightDict")):
            surface += nfiles * self.get_surface(**i)
        self.surface = surface
        self.data = [infile]

    def get_surface(
        self,
        cylinder_height,
        cylinder_radius,
        max_energy,
        max_zenith,
        min_energy,
        min_zenith,
        n_events,
        power_law_index,
        primary_type,
        **kwargs,
    ):
        surface = VolumeCorrCylinder(
            cylinder_height, cylinder_radius, np.cos(max_zenith), np.cos(min_zenith)
        )
        assert power_law_index > 0
        powerlaw = PowerLaw(-power_law_index, min_energy, max_energy)
        probs = [
            GenerationSurface(primary_type, 0.5 * n_events, powerlaw, surface),
            GenerationSurface(-primary_type, 0.5 * n_events, powerlaw, surface),
        ]
        return GenerationSurfaceCollection(*probs)

    def read_weight_map(self, table):
        return [
            dict(
                n_events=get_constant_column(get_column(table, "NEvents")),
                primary_type=get_constant_column(abs(get_column(table, "PrimaryNeutrinoType"))),
                cylinder_height=get_constant_column(get_column(table, "CylinderHeight")),
                cylinder_radius=get_constant_column(get_column(table, "CylinderRadius")),
                min_energy=10 ** get_constant_column(get_column(table, "MinEnergyLog")),
                max_energy=10 ** get_constant_column(get_column(table, "MaxEnergyLog")),
                min_zenith=get_constant_column(get_column(table, "MinZenith")),
                max_zenith=get_constant_column(get_column(table, "MaxZenith")),
                power_law_index=get_constant_column(get_column(table, "PowerLawIndex")),
            )
        ]

    def get_surface_params(self):
        return dict(
            energy=self.get_column("I3MCWeightDict", "PrimaryNeutrinoEnergy"),
            particle_type=self.get_column("I3MCWeightDict", "PrimaryNeutrinoType"),
            cos_zen=np.cos(self.get_column("I3MCWeightDict", "PrimaryNeutrinoZenith")),
        )

    def get_flux_params(self):
        return self.get_surface_params()

    def get_event_weight(self):
        return self.get_column("I3MCWeightDict", "TotalWeight")
