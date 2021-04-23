import numpy as np

from .generation_surface import GenerationSurface
from .powerlaw import PowerLaw
from .spatial import CircleInjector, UniformSolidAngleCylinder
from .utils import Null, constcol, get_column, get_table, has_column
from .weighter import Weighter


def nugen_spatial(table):
    """
    Inspect the `I3MCWeightDict` table object of a nugen file to generate an object to represent
    the spatial distribution. It will either return a CircleInjector or UniformSolidAngleCylinder
    depending on how the dataset was generated
    """

    max_cos = np.cos(constcol(table, "MinZenith"))
    min_cos = np.cos(constcol(table, "MaxZenith"))

    # older nugen used circle injection mode this can be determied by checking `InjectionSurfaceR` it will
    # be > 0 for circle injection and -1 for surface injection. In new versions it is not even present
    # indicating surface mode
    if has_column(table, "InjectionSurfaceR"):
        injection_radius = constcol(table, "InjectionSurfaceR")
    else:
        injection_radius = -1

    if injection_radius > 0:
        return CircleInjector(injection_radius, min_cos, max_cos)

    # CylinderHeight and CylinderRadius were added after the introduction of surface mode
    # os if they are not there they are probabally ( 1900, 950 )
    if has_column(table, "CylinderHeight"):
        cylinder_height = constcol(table, "CylinderHeight")
    else:
        cylinder_height = 1900
    if has_column(table, "CylinderRadius"):
        cylinder_radius = constcol(table, "CylinderRadius")
    else:
        cylinder_radius = 950
    return UniformSolidAngleCylinder(cylinder_height, cylinder_radius, min_cos, max_cos)


def nugen_spectrum(table):
    """
    Inspect the `I3MCWeightDict` table object of a nugen file to generate a PowerLaw object to represent
    the energy spectrum
    """
    min_energy = 10 ** constcol(table, "MinEnergyLog")
    max_energy = 10 ** constcol(table, "MaxEnergyLog")
    # the energy spectrum is always powerlaw however nugen uses positive value of `PowerLawIndex`
    # for negitive slopes ie +2 means E**-2 injection spectrum
    power_law_index = -constcol(table, "PowerLawIndex")
    assert power_law_index <= 0
    return PowerLaw(power_law_index, min_energy, max_energy)


def nugen_surface(table):
    """
    Inspect the `I3MCWeightDict` table object of a nugen file to generate a surface object
    """
    spatial = nugen_spatial(table)
    spectrum = nugen_spectrum(table)
    pdgids = np.unique(get_column(table, "PrimaryNeutrinoType"))

    surfaces = Null()
    for pid in pdgids:
        mask = pid == get_column(table, "PrimaryNeutrinoType")

        # neutrino-generator is usually produced with approximatly equal porportions of nu and nubar
        # newer version will explicitly put the ratio in `TypeWeight` but for older version we
        # assume it is 0.5
        if has_column(table, "TypeWeight"):
            type_weight = constcol(table, "TypeWeight", mask)
        else:
            type_weight = 0.5

        primary_type = constcol(table, "PrimaryNeutrinoType", mask)
        n_events = type_weight * constcol(table, "NEvents", mask)
        surfaces += GenerationSurface(primary_type, n_events, spectrum, spatial)
    return surfaces


def NuGenWeighter(infile, nfiles):
    # pylint: disable=invalid-name
    """
    Weighter for neutrino-generator (NuGen) simulation

    Does not use S-Frames and stores the surface information in an I3MapStringDouble so that the user
    does not know how many jobs contributed to the current sample, so it needs to know the number of
    files. Nugen calculates the event weight in a column called ``TotalWeight`` which takes into account
    the netutrino cross-section, detector density, and distance traveled through the generation volume.
    """

    weight_table = get_table(infile, "I3MCWeightDict")
    surface = nfiles * nugen_surface(weight_table)

    event_map = dict(
        energy=("I3MCWeightDict", "PrimaryNeutrinoEnergy"),
        pdgid=("I3MCWeightDict", "PrimaryNeutrinoType"),
        zenith=("I3MCWeightDict", "PrimaryNeutrinoZenith"),
    )
    # the event weight is stored in `TotalWeight` in newer simulation and
    # `TotalInteractionProbabilityWeight` in older simulation, so we are gonna need to let the weighter
    # know that
    if has_column(weight_table, "TotalWeight"):
        event_map["event_weight"] = ("I3MCWeightDict", "TotalWeight")
    else:
        assert has_column(weight_table, "TotalInteractionProbabilityWeight")
        event_map["event_weight"] = ("I3MCWeightDict", "TotalInteractionProbabilityWeight")

    return Weighter([infile], surface, event_map)
