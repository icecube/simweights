#!/usr/bin/env python
from icecube.weighting import fluxes as oldfluxes 
from icecube.simweights import fluxes as newfluxes
from icecube.corsika_reader import CorsikaToPDG
import numpy as np
import unittest

class TestCosmicRayModels(unittest.TestCase):
    def flux_cmp(self,name):
        f1 = getattr(oldfluxes,name)()
        f2 = getattr(newfluxes,name)()
        N=1000
        E=np.logspace(2,10,N)
        ptypes = [ CorsikaToPDG(p) for p in f1.ptypes]
        self.assertEqual(ptypes,f2.ptypes)
        for p in ptypes:
            v1=f1(E,p)
            v2=f2(E,p)
            for i in range(N):
                self.assertAlmostEqual(v1[i],v2[i],17)
                if v2[i]!=0:
                    self.assertAlmostEqual(v1[i]/v2[i],1,12)


    def test_fixed_fractional_flux(self):
        """
        The old FixedFractionalFlux needed some hand holding
        so do it in a sperate test
        """
        f= { 2212:.1,1000020040:.2,1000080160:.3,1000260560:.4}
        f1 = oldfluxes.FixedFractionFlux(f)
        f2 = newfluxes.FixedFractionFlux(f)
        N=1000
        E=np.logspace(2,10,N)
        self.assertEqual(f1.ptypes,f2.ptypes)
        for p in f1.ptypes:
            v1=[f1(EE,p) for EE in E]
            v2=f2(E,p)
            for i in range(N):
                self.assertAlmostEqual(v1[i],v2[i],17)
                if v2[i]!=0:
                    self.assertAlmostEqual(v1[i]/v2[i],1,12)

models=['GaisserH3a','GaisserH4a','GaisserH4a_IT','GaisserHillas','GlobalFitGST',
        'Hoerandel','Hoerandel5','Hoerandel_IT','Honda2004','TIG1996']

for m in models:
    setattr(TestCosmicRayModels,'test_'+m,lambda self,m=m: self.flux_cmp(m))

if __name__ == '__main__':
    unittest.main()
