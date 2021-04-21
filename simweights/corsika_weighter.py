import numpy as np

from .cylinder import NaturalRateCylinder
from .utils import get_column, get_constant_column, get_table
from .weighter import MapWeighter


class CorsikaWeighter(MapWeighter):
    """
    Weighter for CORSIKA-in-ice simulation made with I3CORSIKAReader

    I3CORSIKAReader not use S-Frames and stores the surface information in an I3MapStringDouble so that
    the user does not know how many jobs contributed to the current sample, so it needs to know the number
    of files.
    """

    _spatial_distribution = NaturalRateCylinder

    @staticmethod
    def _get_surface_map(infile):
        table = get_table(infile, "CorsikaWeightMap")
        vals = {}
        vals["ParticleType"] = sorted(np.unique(get_column(table, "ParticleType").astype(int)))
        for name in ("CylinderLength", "CylinderRadius", "ThetaMin", "ThetaMax", "OverSampling", "Weight"):
            vals[name] = get_constant_column(get_column(table, name))

        for i in range(len(vals["ParticleType"])):
            mask = vals["ParticleType"][i] == get_column(table, "ParticleType")
            for name in ("NEvents", "EnergyPrimaryMax", "EnergyPrimaryMin", "PrimarySpectralIndex"):
                if name not in vals:
                    vals[name] = []
                vals[name].append(get_constant_column(get_column(table, name)[mask]))

        return [
            dict(
                primary_type=p,
                n_events=vals["OverSampling"] * vals["NEvents"][i],
                cylinder_height=vals["CylinderLength"],
                cylinder_radius=vals["CylinderRadius"],
                min_zenith=vals["ThetaMin"],
                max_zenith=vals["ThetaMax"],
                min_energy=vals["EnergyPrimaryMin"][i],
                max_energy=vals["EnergyPrimaryMax"][i],
                power_law_index=round(vals["PrimarySpectralIndex"][i], 6),
            )
            for i, p in enumerate(vals["ParticleType"])
        ]

    def _get_surface_params(self):
        return dict(
            pdgid=self.get_column("PolyplopiaPrimary", "type"),
            energy=self.get_column("PolyplopiaPrimary", "energy"),
            cos_zen=np.cos(self.get_column("PolyplopiaPrimary", "zenith")),
        )

    def _get_flux_params(self):
        return dict(
            pdgid=self.get_column("PolyplopiaPrimary", "type"),
            energy=self.get_column("PolyplopiaPrimary", "energy"),
        )

    def _get_event_weight(self):
        return np.ones_like(self.get_column("PolyplopiaPrimary", "energy"))
