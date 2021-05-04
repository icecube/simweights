import warnings

import numpy as np

from .generation_surface import GenerationSurface
from .powerlaw import PowerLaw
from .spatial import NaturalRateCylinder
from .utils import constcol, get_column, get_table, has_table
from .weighter import Weighter


def sframe_corsika_surface(table, oversampling):
    """
    Inspect the rows of a CORSIKA S-Frame table object to generate a surface object. This function works
    on files generated with either triggered CORSIKA or corsika-reader because `I3PrimaryInjectorInfo` and
    `I3CorsikaInfo` use exactly the same names for quantities.
    """
    surfaces = []
    for row in table:
        assert row["power_law_index"] <= 0
        spatial = NaturalRateCylinder(
            row["cylinder_height"],
            row["cylinder_radius"],
            np.cos(row["max_zenith"]),
            np.cos(row["min_zenith"]),
        )
        spectrum = PowerLaw(row["power_law_index"], row["min_energy"], row["max_energy"])
        if oversampling:
            oversampling_val = row["oversampling"]
        else:
            oversampling_val = 1
        surfaces.append(
            row["n_events"] * oversampling_val * GenerationSurface(row["primary_type"], spectrum, spatial)
        )
    return sum(surfaces)


def weight_map_corsika_surface(table):
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


def CorsikaWeighter(infile, nfiles=None):
    # pylint: disable=invalid-name
    """
    Weighter for CORSIKA-in-ice simulation made with I3CORSIKAReader

    I3CORSIKAReader not use S-Frames and stores the surface information in an I3MapStringDouble so that
    the user does not know how many jobs contributed to the current sample, so it needs to know the number
    of files.
    """

    if has_table(infile, "I3CorsikaWeight"):
        if nfiles is not None:
            raise RuntimeError(
                "This file, `{}`, was identified as a triggered CORSIKA file based on the existance of "
                "the `I3CorsikaWeight` object. However the parameter nfiles was passed to CorsikaWeighter."
                "This is unnsecessary.".format(getattr(infile, "filename", "<NONE>"))
            )

        info_obj = "I3PrimaryInjectorInfo"
        if not has_table(infile, info_obj):
            raise RuntimeError(
                "This file, `{}`, was identified as a triggered CORSIKA file based on the existance of "
                "the `I3CorsikaWeight` object. However it seems to be missing the S-Frames table "
                "`I3PrimaryInjectorInfo`.".format(getattr(infile, "filename", "<NONE>"))
            )
        event_map = dict(
            pdgid=("I3CorsikaWeight", "type"),
            energy=("I3CorsikaWeight", "energy"),
            zenith=("I3CorsikaWeight", "zenith"),
            event_weight=("I3CorsikaWeight", "weight"),
        )
        surface = sframe_corsika_surface(get_table(infile, info_obj), oversampling=False)

    elif nfiles is None:

        info_obj = "I3CorsikaInfo"
        if not has_table(infile, info_obj):
            raise RuntimeError(
                "File `{}` is was not passed an parameter for nfiles and no I3CorsikaInfo table was "
                " found.".format(getattr(infile, "filename", "<NONE>"))
            )
        event_map = dict(
            pdgid=("PolyplopiaPrimary", "type"),
            energy=("PolyplopiaPrimary", "energy"),
            zenith=("PolyplopiaPrimary", "zenith"),
            event_weight=None,
        )
        surface = sframe_corsika_surface(get_table(infile, info_obj), oversampling=True)

    else:

        if not np.isscalar(nfiles):
            raise TypeError("CorsikaWeighter: Nfiles must be an integer. Got " + str(nfiles))

        if has_table(infile, "I3CorsikaInfo"):
            warnings.warn(
                "CorsikaWeighter was given a value for nfiles, but this file has an "
                "I3CorsikaInfo table indicating it has S-Frames"
            )

        event_map = dict(
            energy=("PolyplopiaPrimary", "energy"),
            pdgid=("PolyplopiaPrimary", "type"),
            zenith=("PolyplopiaPrimary", "zenith"),
            event_weight=None,
        )
        table = get_table(infile, "CorsikaWeightMap")
        surface = nfiles * weight_map_corsika_surface(table)

    return Weighter([infile], surface, event_map)
