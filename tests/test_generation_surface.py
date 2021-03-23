#!/usr/bin/env python
import numpy as np
from simweights import PDGCode, PowerLaw, VolumeCorrCylinder, GenerationSurface, GenerationSurfaceCollection
import unittest

class TestGenerationSurface(unittest.TestCase):

    def setUp(self):
        self.p1 = PowerLaw(-1, 10, 100)
        self.p2 = PowerLaw(-2, 50, 500)
        self.c1 = VolumeCorrCylinder(3, 8, -1, 1)
        self.c2 = VolumeCorrCylinder(4, 8, -1, 1)

        self.s0 = GenerationSurface(2212, 10000, self.p1, self.c1)
        self.s1 = GenerationSurface(2212, 20000, self.p1, self.c1)
        self.s2 = GenerationSurface(2212, 10000, self.p2, self.c1)
        self.s3 = GenerationSurface(2212, 10000, self.p1, self.c2)
        self.s4 = GenerationSurface(2213, 10000, self.p1, self.c1)

        self.gsc1 = GenerationSurfaceCollection([self.s0,self.s1])
        self.gsc2 = GenerationSurfaceCollection([self.s0,self.s2])
        self.gsc3 = GenerationSurfaceCollection([self.s0,self.s3])
        self.gsc4 = GenerationSurfaceCollection([self.s0,self.s4])

    def test_compatible(self):
        assert(self.s0.is_compatible(self.s0))
        assert(self.s1.is_compatible(self.s1))
        assert(self.s2.is_compatible(self.s2))
        assert(self.s3.is_compatible(self.s3))
        assert(self.s4.is_compatible(self.s4))

        assert(self.s0.is_compatible(self.s1))
        assert(not self.s0.is_compatible(self.s2))
        assert(not self.s0.is_compatible(self.s3))
        assert(not self.s0.is_compatible(self.s4))

        self.assertEqual(self.s0,self.s0)
        self.assertEqual(self.s1,self.s1)
        self.assertEqual(self.s2,self.s2)
        self.assertEqual(self.s3,self.s3)
        self.assertEqual(self.s4,self.s4)

        self.assertNotEqual(self.s0,self.s1)
        self.assertNotEqual(self.s0,self.s2)
        self.assertNotEqual(self.s0,self.s3)
        self.assertNotEqual(self.s0,self.s4)

    def test_addition(self):
        n0 = self.s0.nevents
        n1 = self.s1.nevents
        s = self.s0 + self.s1
        self.assertEqual(type(s),GenerationSurface)
        self.assertEqual(s.nevents,30000)
        self.assertEqual(s.nevents,self.s0.nevents+self.s1.nevents)
        self.assertNotEqual(s,self.s0)
        self.assertNotEqual(s,self.s1)
        self.assertEqual(n0,self.s0.nevents)
        self.assertEqual(n1,self.s1.nevents)
        self.assertAlmostEqual(
            s.get_extended_pdf(2212,50,0),
            self.s0.get_extended_pdf(2212,50,0) + self.s1.get_extended_pdf(2212,50,0)
        )

        ss = self.s0 + self.s2
        self.assertEqual(type(ss), GenerationSurfaceCollection)
        self.assertEqual(len(ss.spectra),1)
        self.assertEqual(len(ss.spectra[2212]),2)
        self.assertEqual(ss.spectra[2212][0], self.s0)
        self.assertEqual(ss.spectra[2212][1], self.s2)
        self.assertAlmostEqual(
            ss.get_extended_pdf(2212,50,0),
            self.s0.get_extended_pdf(2212,50,0) + self.s2.get_extended_pdf(2212,50,0)
        )

        s3 = self.s0 + self.s3
        self.assertEqual(type(s3), GenerationSurfaceCollection)
        self.assertEqual(len(s3.spectra),1)
        self.assertEqual(len(s3.spectra[2212]),2)
        self.assertEqual(s3.spectra[2212][0], self.s0)
        self.assertEqual(s3.spectra[2212][1], self.s3)
        self.assertAlmostEqual(
            s3.get_extended_pdf(2212,50,0),
            self.s0.get_extended_pdf(2212,50,0) + self.s3.get_extended_pdf(2212,50,0)
        )

        s4 = self.s0 + self.s4
        self.assertEqual(type(s4), GenerationSurfaceCollection)
        self.assertEqual(len(s4.spectra),2)
        self.assertEqual(len(s4.spectra[2212]),1)
        self.assertEqual(len(s4.spectra[2213]),1)
        self.assertEqual(s4.spectra[2212][0], self.s0)
        self.assertEqual(s4.spectra[2213][0], self.s4)
        self.assertAlmostEqual(s4.get_extended_pdf(2212,50,0),self.s0.get_extended_pdf(2212,50,0))
        self.assertAlmostEqual(s4.get_extended_pdf(2213,50,0),self.s4.get_extended_pdf(2213,50,0))

        with self.assertRaises(TypeError):
            print(self.s0 + None)

        with self.assertRaises(TypeError):
            print(self.s0 + int)

        with self.assertRaises(TypeError):
            print(self.s0 + PowerLaw)

    def test_multiplication(self):
        sa = self.s0
        sa *= 4.4
        self.assertEqual(sa.nevents,44000)
        self.assertEqual(self.s0.nevents, 10000)
        self.assertAlmostEqual(sa.get_extended_pdf(2212, 50, 0),
                               4.4 * self.s0.get_extended_pdf(2212, 50, 0))

        sb = self.s0 * 5.5
        self.assertEqual(sb.nevents,55000)
        self.assertEqual(self.s0.nevents, 10000)
        self.assertAlmostEqual(sb.get_extended_pdf(2212, 50, 0),
                               5.5 * self.s0.get_extended_pdf(2212, 50, 0))

        sc = 6.6 * self.s0
        self.assertEqual(sc.nevents,66000)
        self.assertEqual(self.s0.nevents, 10000)
        self.assertAlmostEqual(sc.get_extended_pdf(2212, 50, 0),
                               6.6 * self.s0.get_extended_pdf(2212, 50, 0))

    def test_repr(self):
        rep = str(self.s0)
        print(rep)
        self.assertEqual(rep[:18],"GenerationSurface(")
        self.assertEqual(rep[-1:],")")
        v = rep[18:-1].split(',')
        print (v)
        self.assertEqual(v[0],'PPlus')
        self.assertEqual(float(v[1]),1e4)
        self.assertEqual(','.join(v[2:5]).strip(),repr(self.p1))
        self.assertEqual(','.join(v[5:9]).strip(),repr(self.c1))

    def test_powerlaw(self):
        N=self.s0.nevents
        E = np.geomspace(self.p1.a, self.p1.b - 1 / N, N)
        cz = np.linspace(self.c1.a, self.c1.b, N)
        w = 1/self.s0.get_extended_pdf(2212, E, cz)

        area = (self.p1.b - self.p1.a) * (2 * self.c1.radius * np.pi**2
                                          * (self.c1.radius + self.c1.length))

        self.assertAlmostEqual(area, self.s0.get_surface_area())
        self.assertAlmostEqual(area, self.s0.spectrum.span * self.s0.surface.etendue)
        self.assertAlmostEqual(w.sum()/area, 1, 4)

        self.assertEqual(self.s0.surface,self.c1)
        self.assertIsNot(self.s0.surface,self.c1)
        self.assertEqual(self.s0.spectrum,self.p1)
        self.assertIsNot(self.s0.spectrum,self.p1)

    def test_two_surfaces(self):
        N=self.s0.nevents
        cz = np.linspace(self.c1.a, self.c1.b, N)
        q = np.linspace(1 / 2 / N, 1 - 1 / 2 / N, N)
        E1  = 10 * np.exp(q * np.log(100 / 10))
        w1 = 1 / self.s0.get_extended_pdf(2212, E1, cz)
        E2 = (q * (500**-1 - 50**-1)  + 50**-1)**-1
        w2 = 1 / self.s2.get_extended_pdf(2212, E2, cz)

        surf = self.s0 + self.s2
        E = np.r_[E1, E2]
        czc = np.r_[cz, cz]
        wc = 1/surf.get_extended_pdf(2212, E, czc)

        self.assertAlmostEqual(wc.sum() / (self.p2.b - self.p1.a) / self.c1.etendue, 1, 3)

        self.assertEqual(self.s0.surface, self.c1)
        self.assertIsNot(self.s0.surface, self.c1)
        self.assertEqual(self.s0.spectrum, self.p1)
        self.assertIsNot(self.s0.spectrum, self.p1)

        self.assertEqual(self.s2.surface, self.c1)
        self.assertIsNot(self.s2.surface, self.c1)
        self.assertEqual(self.s2.spectrum, self.p2)
        self.assertIsNot(self.s2.spectrum, self.p2)

        self.assertEqual(len(surf.spectra), 1)
        np.testing.assert_array_equal(list(surf.spectra.keys()), [2212])
        
        self.assertEqual(surf.spectra[2212][0],self.s0)
        self.assertIsNot(surf.spectra[2212][0],self.s0)
        self.assertEqual(surf.spectra[2212][0].surface,self.s0.surface)
        self.assertIsNot(surf.spectra[2212][0].surface,self.s0.surface)
        self.assertEqual(surf.spectra[2212][0].spectrum,self.s0.spectrum)
        self.assertIsNot(surf.spectra[2212][0].spectrum,self.s0.spectrum)      
        
        self.assertEqual(surf.spectra[2212][1],self.s2)
        self.assertIsNot(surf.spectra[2212][1],self.s2)
        self.assertEqual(surf.spectra[2212][1].surface, self.s2.surface)
        self.assertIsNot(surf.spectra[2212][1].surface, self.s2.surface)
        self.assertEqual(surf.spectra[2212][1].spectrum, self.s2.spectrum)
        self.assertIsNot(surf.spectra[2212][1].spectrum, self.s2.spectrum)

    def test_instantiation(self):
        s02 = GenerationSurfaceCollection([self.s0,self.s0])
        self.assertEqual(len(s02.spectra),1)
        self.assertEqual(len(s02.spectra[2212]),1)
        assert s02.spectra[2212][0].is_compatible(self.s0)
        self.assertEqual(s02.spectra[2212][0].nevents,20000)

        s02 = GenerationSurfaceCollection([self.s0,self.s2])
        self.assertEqual(len(s02.spectra),1)
        self.assertEqual(len(s02.spectra[2212]),2)
        assert s02.spectra[2212][0].is_compatible(self.s0)
        assert s02.spectra[2212][1].is_compatible(self.s2)
        self.assertEqual(s02.spectra[2212][0].nevents,10000)
        self.assertEqual(s02.spectra[2212][1].nevents,10000)

        s04 = GenerationSurfaceCollection([self.s0,self.s4])
        self.assertEqual(len(s04.spectra),2)
        self.assertEqual(len(s04.spectra[2212]),1)
        self.assertEqual(len(s04.spectra[2213]),1)
        assert s04.spectra[2212][0].is_compatible(self.s0)
        assert s04.spectra[2213][0].is_compatible(self.s4)
        self.assertEqual(s04.spectra[2212][0].nevents,10000)
        self.assertEqual(s04.spectra[2213][0].nevents,10000)

if __name__ == '__main__':
    unittest.main()

