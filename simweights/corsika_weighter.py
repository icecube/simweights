import numpy as np

from .generation_surface import GenerationSurface
from .powerlaw import PowerLaw
from .spatial import NaturalRateCylinder
from .utils import Null, constcol, get_column, get_table
from .weighter import Weighter


class CorsikaWeighter(Weighter):
    """
    Weighter for CORSIKA-in-ice simulation made with I3CORSIKAReader

    I3CORSIKAReader not use S-Frames and stores the surface information in an I3MapStringDouble so that
    the user does not know how many jobs contributed to the current sample, so it needs to know the number
    of files.
    """

    event_map = dict(
        energy=("PolyplopiaPrimary", "energy"),
        pdgid=("PolyplopiaPrimary", "type"),
        zenith=("PolyplopiaPrimary", "zenith"),
        event_weight=None,
    )

    def __init__(self, infile, nfiles):
        surface = nfiles * self._get_surface(infile)
        super().__init__(surface, [infile])

    @staticmethod
    def _get_surface(infile):
        table = get_table(infile, "CorsikaWeightMap")
        pdgids = sorted(np.unique(get_column(table, "ParticleType").astype(int)))
        surface = Null()
        for pdgid in pdgids:
            mask = pdgid == get_column(table, "ParticleType")

            def gcol(name):
                return constcol(get_column(table, name)[mask])

            spatial = NaturalRateCylinder(
                gcol("CylinderLength"),
                gcol("CylinderRadius"),
                np.cos(gcol("ThetaMax")),
                np.cos(gcol("ThetaMin")),
            )

            primary_spectral_index = round(constcol(get_column(table, "PrimarySpectralIndex")[mask]), 6)
            assert primary_spectral_index <= 0

            spectrum = PowerLaw(primary_spectral_index, gcol("EnergyPrimaryMin"), gcol("EnergyPrimaryMax"))
            nevents = gcol("OverSampling") * gcol("NEvents")
            surface += GenerationSurface(pdgid, nevents, spectrum, spatial)

        return surface
