import numpy as np

from .generation_surface import GenerationSurface
from .powerlaw import PowerLaw
from .spatial import NaturalRateCylinder
from .utils import constcol, get_column, get_table
from .weighter import Weighter


def corsika_surface(table):
    """
    Inspect the `CorsikaWeightMap` table object of a corsika file to generate a surface object

    """
    pdgids = sorted(np.unique(get_column(table, "ParticleType").astype(int)))
    surface = []
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
        surface.append(nevents * GenerationSurface(pdgid, spectrum, spatial))

    return sum(surface)


def CorsikaWeighter(infile, nfiles):
    # pylint: disable=invalid-name
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
    table = get_table(infile, "CorsikaWeightMap")
    surface = nfiles * corsika_surface(table)
    return Weighter([infile], surface, event_map)

    # def __init__(self, infile, nfiles):
    #    surface = nfiles * self._get_surface(infile)
    #    super().__init__(surface, [infile])
