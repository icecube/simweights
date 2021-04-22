import numpy as np

from .cylinder import UniformSolidAngleCylinder
from .generation_surface import GenerationSurface
from .powerlaw import PowerLaw
from .utils import Null, constcol, get_column, get_table, has_column
from .weighter import Weighter


class NuGenWeighter(Weighter):
    """
    Weighter for neutrino-generator (NuGen) simulation

    Does not use S-Frames and stores the surface information in an I3MapStringDouble so that the user
    does not know how many jobs contributed to the current sample, so it needs to know the number of
    files. Nugen calculates the event weight in a column called ``TotalWeight`` which takes into account
    the netutrino cross-section, detector density, and distance traveled through the generation volume.
    """

    event_map = dict(
        energy=("I3MCWeightDict", "PrimaryNeutrinoEnergy"),
        pdgid=("I3MCWeightDict", "PrimaryNeutrinoType"),
        zenith=("I3MCWeightDict", "PrimaryNeutrinoZenith"),
        event_weight=("I3MCWeightDict", "TotalWeight"),
    )

    def __init__(self, infile, nfiles, cylinder_height=None, cylinder_radius=None):
        self.cylinder_height = cylinder_height
        self.cylinder_radius = cylinder_radius
        surface = nfiles * self._get_surface(infile)
        super().__init__(surface, [infile])

    def _get_surface(self, infile):
        # nugen generates both nu and nubar with the fraction stored in TypeWeight and the total stored
        # in NEvents
        table = get_table(infile, "I3MCWeightDict")

        min_zenith = constcol(get_column(table, "MinZenith"))
        max_zenith = constcol(get_column(table, "MaxZenith"))
        if has_column(table, "InjectionRadius"):
            # older data
            raise NotImplementedError()
        else:
            if has_column(table, "CylinderHeight"):
                cylinder_height = constcol(get_column(table, "CylinderHeight"))
            else:
                cylinder_height = self.cylinder_height if self.cylinder_height is not None else 1900
            if has_column(table, "CylinderRadius"):
                cylinder_radius = constcol(get_column(table, "CylinderRadius"))
            else:
                cylinder_radius = self.cylinder_radius if self.cylinder_radius is not None else 950
            spatial = UniformSolidAngleCylinder(
                cylinder_height, cylinder_radius, np.cos(max_zenith), np.cos(min_zenith)
            )

        min_energy = 10 ** constcol(get_column(table, "MinEnergyLog"))
        max_energy = 10 ** constcol(get_column(table, "MaxEnergyLog"))
        power_law_index = -constcol(get_column(table, "PowerLawIndex"))
        spectrum = PowerLaw(power_law_index, min_energy, max_energy)

        pdgids = np.unique(get_column(table, "PrimaryNeutrinoType"))
        surfaces = Null()
        for pid in pdgids:
            mask = pid == get_column(table, "PrimaryNeutrinoType")
            primary_type = constcol(get_column(table, "PrimaryNeutrinoType")[mask])
            n_events = constcol(get_column(table, "TypeWeight")[mask]) * constcol(
                get_column(table, "NEvents")[mask]
            )
            surfaces += GenerationSurface(primary_type, n_events, spectrum, spatial)
        return surfaces
