#!/usr/bin/env python
import numpy as np
from simweights import PDGCode,PowerLaw,VolumeCorrCylinder,GenerationSurface
import unittest

class TestGenerationSurface(unittest.TestCase):

    def test_powerlaw(self):

        N=10000
        ptype = PDGCode.PPlus
        spectrum = PowerLaw(-1, 10, 100)
        fullcylinder = VolumeCorrCylinder(3, 8, -1, 1)

        s0 = GenerationSurface(ptype, N, spectrum, fullcylinder)
        E = np.geomspace(spectrum.a + 1 / 2 / N, spectrum.b - 1 / 2 / N, N)
        cz = np.linspace(fullcylinder.a, fullcylinder.b, N)
        w = 1/s0.get_extended_pdf(ptype, E, cz)
        
        area = (spectrum.b - spectrum.a) * (2 * fullcylinder.radius * np.pi**2
                                            * (fullcylinder.radius + fullcylinder.length))

        self.assertAlmostEqual(area, s0.get_surface_area())
        self.assertAlmostEqual(area, s0.spectrum.span * s0.surface.etendue)        
        self.assertAlmostEqual(w.sum()/area, 1, 4)

        self.assertEqual(s0.surface,fullcylinder)
        self.assertIsNot(s0.surface,fullcylinder)
        self.assertEqual(s0.spectrum,spectrum)
        self.assertIsNot(s0.spectrum,spectrum)

    def test_compatible(self):
        p1 = PowerLaw(-1, 10, 100)
        p2 = PowerLaw(-2, 10, 100)
        c1 = VolumeCorrCylinder(3, 8, -1, 1)
        c2 = VolumeCorrCylinder(4, 8, -1, 1)

        s0 = GenerationSurface(2212, 100, p1, c1)
        s1 = GenerationSurface(2212, 200, p1, c1)
        s2 = GenerationSurface(2212, 100, p2, c1)
        s3 = GenerationSurface(2212, 100, p1, c2)
        s4 = GenerationSurface(2213, 100, p1, c1)

        assert(s0.is_compatible(s0))
        assert(s1.is_compatible(s1))
        assert(s2.is_compatible(s2))
        assert(s3.is_compatible(s3))
        assert(s4.is_compatible(s4))

        assert(s0.is_compatible(s1))
        assert(not s0.is_compatible(s2))
        assert(not s0.is_compatible(s3))
        assert(not s0.is_compatible(s4))

        self.assertEqual(s0,s0)
        self.assertEqual(s1,s1)
        self.assertEqual(s2,s2)
        self.assertEqual(s3,s3)
        self.assertEqual(s4,s4)

        self.assertNotEqual(s0,s1)
        self.assertNotEqual(s0,s2)
        self.assertNotEqual(s0,s3)
        self.assertNotEqual(s0,s4)


    def test_two_surfaces(self):

        N=10000
        ptype = PDGCode.PPlus
        cyl = VolumeCorrCylinder(3, 8, -1, 1)
        cz = np.linspace(cyl.a, cyl.b, N)

        spec1 = PowerLaw(-1, 10, 100)
        surf1 = GenerationSurface(ptype, N, spec1, cyl)
        E1 = np.geomspace(spec1.a+1/2/N, spec1.b - 1 / 2 / N, N)
        w1 = 1 / surf1.get_extended_pdf(ptype, E1, cz)

        spec2 = PowerLaw(-2, 50, 500)
        surf2 = GenerationSurface(ptype, N, spec2, cyl)
        E2 = spec2.ppf(np.linspace(1 / 2 / N, 1 - 1 / 2 / N, N))
        w2 = 1 / surf2.get_extended_pdf(ptype, E2, cz)

        surf = surf1 + surf2
        E = np.r_[E1, E2]
        czc = np.r_[cz, cz]
        wc = 1/surf.get_extended_pdf(ptype, E, czc)

        self.assertAlmostEqual(wc.sum() / (spec2.b - spec1.a) / cyl.etendue, 1, 3)

        self.assertEqual(surf1.surface, cyl)
        self.assertIsNot(surf1.surface, cyl)
        self.assertEqual(surf1.spectrum, spec1)
        self.assertIsNot(surf1.spectrum, spec1)

        self.assertEqual(surf2.surface, cyl)
        self.assertIsNot(surf2.surface, cyl)
        self.assertEqual(surf2.spectrum, spec2)
        self.assertIsNot(surf2.spectrum, spec2)

        self.assertEqual(len(surf.spectra), 1)
        np.testing.assert_array_equal(list(surf.spectra.keys()), [2212])
        
        self.assertEqual(surf.spectra[2212][0],surf1)
        self.assertIsNot(surf.spectra[2212][0],surf1)
        self.assertEqual(surf.spectra[2212][0].surface,surf1.surface)
        self.assertIsNot(surf.spectra[2212][0].surface,surf1.surface)
        self.assertEqual(surf.spectra[2212][0].spectrum,surf1.spectrum)
        self.assertIsNot(surf.spectra[2212][0].spectrum,surf1.spectrum)                
        
        self.assertEqual(surf.spectra[2212][1],surf2)
        self.assertIsNot(surf.spectra[2212][1],surf2)        
        self.assertEqual(surf.spectra[2212][1].surface, surf2.surface)
        self.assertIsNot(surf.spectra[2212][1].surface, surf2.surface)
        self.assertEqual(surf.spectra[2212][1].spectrum, surf2.spectrum)
        self.assertIsNot(surf.spectra[2212][1].spectrum, surf2.spectrum)

if __name__ == '__main__':
    unittest.main()

