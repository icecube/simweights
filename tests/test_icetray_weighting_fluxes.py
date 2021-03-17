#!/usr/bin/env python
try:
    from icecube.corsika_reader import CorsikaToPDG,PDGToCorsika,CorsikaToPDG
    from icecube.weighting import fluxes as oldfluxes
except ImportError:
    oldfluxes=False

from simweights import fluxes as newfluxes
import numpy as np
import unittest

@unittest.skipUnless(oldfluxes, "IceTray libraries not found")
class TestCosmicRayModels(unittest.TestCase):
    def flux_cmp(self, name):
        f1 = getattr(oldfluxes, name)()
        f2 = getattr(newfluxes, name)()
        N = 1000
        E = np.logspace(2, 10, N)
        self.assertEqual([CorsikaToPDG(p) for p in f1.ptypes], f2.ptypes)

        if isinstance(f1, oldfluxes.CompiledFlux):
            f1._translator = None
            ptypes = f1.ptypes
        else:
            ptypes = f2.ptypes
        for j in range(len(ptypes)):
            v1=f1(E, ptypes[j])
            v2=f2(E, f2.ptypes[j])
            for i in range(len(E)):
                self.assertAlmostEqual(v1[i], v2[i], 17)
                if v2[i] != 0:
                    self.assertAlmostEqual(v1[i]/v2[i], 1, 12)

    def test_fixed_fractional_flux(self):
        """
        The old FixedFractionalFlux needed some hand holding
        so do it in a sperate test
        """
        f= { 2212: .1, 1000020040: .2, 1000080160: .3, 1000260560: .4 }
        f1 = oldfluxes.FixedFractionFlux({PDGToCorsika(p):x for p,x in f.items()})
        f2 = newfluxes.FixedFractionFlux(f)
        N = 1000
        E=np.logspace(2, 10, N)
        self.assertEqual([CorsikaToPDG(p) for p in f1.ptypes], f2.ptypes)
        for p in f2.ptypes:
            v1 = [ f1(EE, PDGToCorsika(p)) for EE in E ]
            v2 = f2(E, p)
            for i in range(N):
                self.assertAlmostEqual(v1[i], v2[i], 17)
                if v2[i]!=0:
                    self.assertAlmostEqual(v1[i] / v2[i], 1, 12)


    def test_corsika_to_pdg(self):
        ctypes = [14,  402, 1407, 1608, 2713, 5626]
        assert np.all(oldfluxes.FixedFractionFlux.corsika_to_pdg(ctypes) == newfluxes.corsika_to_pdg(ctypes))


models=['GaisserH3a', 'GaisserH4a', 'GaisserH4a_IT', 'GaisserHillas', 'GlobalFitGST',
        'Hoerandel', 'Hoerandel5', 'Hoerandel_IT', 'Honda2004', 'TIG1996']

for m in models:
    setattr(TestCosmicRayModels, 'test_' + m , lambda self, m=m : self.flux_cmp(m))

if __name__ == '__main__':
    unittest.main()
