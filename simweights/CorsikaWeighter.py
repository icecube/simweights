import numpy as np

from .utils import Null, get_column, get_constant_column, get_table
from .WeighterBase import Weighter


class CorsikaWeighter(Weighter):
    def __init__(self, infile, nfiles):
        assert nfiles is not None
        surface = Null()
        for smap in self.get_surface_map(infile):
            surface += nfiles * self.get_surface(smap)
        super().__init__(surface, [infile])

    @staticmethod
    def get_surface_map(infile):
        table = get_table(infile, "CorsikaWeightMap")
        vals = {}
        vals["ParticleType"] = sorted(np.unique(get_column(table, "ParticleType").astype(int)))
        for x in ("CylinderLength", "CylinderRadius", "ThetaMin", "ThetaMax", "OverSampling", "Weight"):
            vals[x] = get_constant_column(get_column(table, x))

        for i in range(len(vals["ParticleType"])):
            mask = vals["ParticleType"][i] == get_column(table, "ParticleType")
            for x in ("NEvents", "EnergyPrimaryMax", "EnergyPrimaryMin", "PrimarySpectralIndex"):
                if x not in vals:
                    vals[x] = []
                vals[x].append(get_constant_column(get_column(table, x)[mask]))

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

    def get_surface_params(self):
        return dict(
            particle_type=self.get_column("PolyplopiaPrimary", "type"),
            energy=self.get_column("PolyplopiaPrimary", "energy"),
            cos_zen=np.cos(self.get_column("PolyplopiaPrimary", "zenith")),
        )

    def get_flux_params(self):
        return dict(
            ptype=self.get_column("PolyplopiaPrimary", "type"),
            E=self.get_column("PolyplopiaPrimary", "energy"),
        )

    def get_event_weight(self):
        return np.ones_like(self.get_column("PolyplopiaPrimary", "energy"))
