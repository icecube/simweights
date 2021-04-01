#!/usr/bin/env python 
import unittest
import numpy as np
from simweights.cylinder import CylinderBase
from simweights import VolumeCorrCylinder

class TestCylinder(unittest.TestCase):
    def check_diff_etendue(self,c,l,r):
        self.assertAlmostEqual(c.projected_area( -1), np.pi*r**2)
        self.assertAlmostEqual(c.projected_area(-.5), np.pi * r**2 / 2 + 3**.5*l*r )
        self.assertAlmostEqual(c.projected_area(-1/2**.5),r/2**.5*(np.pi *r + 2*l))
        self.assertAlmostEqual(c.projected_area(  0), 2*l*r)
        self.assertAlmostEqual(c.projected_area( 1/2**.5), r/2**.5 *(np.pi*r + 2*l))
        self.assertAlmostEqual(c.projected_area( .5), np.pi * r**2 / 2 + 3**.5*l*r )        
        self.assertAlmostEqual(c.projected_area(  1), np.pi*r**2)

        with self.assertRaises(AssertionError):
            c.projected_area(-1.01)
        with self.assertRaises(AssertionError):
            c.projected_area( 1.01)            
     
        self.assertAlmostEqual(c._diff_etendue(-1), -np.pi**2*r*(r+2*l))
        self.assertAlmostEqual(c._diff_etendue(-.5), -np.pi**2/4 *r* (r + 2/3*l * (3**1.5/np.pi + 8)))        
        self.assertAlmostEqual(c._diff_etendue(-.5**.5), - np.pi**2 / 2 * r * (r + l * (2 / np.pi + 3)))        
        self.assertAlmostEqual(c._diff_etendue( 0), -np.pi**2*l*r )
        self.assertAlmostEqual(c._diff_etendue(.5**.5),  np.pi**2/2 *r* (r + l * (2/np.pi - 1)))        
        self.assertAlmostEqual(c._diff_etendue(.5), np.pi**2/4 * r * (r + 2/3*l * (3**1.5/np.pi - 4)))
        self.assertAlmostEqual(c._diff_etendue( 1), (np.pi*r)**2)        

        with self.assertRaises(AssertionError):
            c._diff_etendue(np.nextafter(-1,-np.inf))
        with self.assertRaises(AssertionError):
            c._diff_etendue(np.nextafter(1,np.inf))

    def check_pdf_etendue(self,c,etendue):
        self.assertAlmostEqual(c.etendue, etendue)

        x = np.linspace(c.cos_zen_min, c.cos_zen_max)
        np.testing.assert_array_almost_equal(c.pdf(x), 1 / etendue)

        x = np.linspace(-2,np.nextafter(c.cos_zen_min, -np.inf))
        np.testing.assert_array_equal(c.pdf(x), 0)

        x = np.linspace(np.nextafter(c.cos_zen_max, np.inf), 2)
        np.testing.assert_array_equal(c.pdf(x), 0)

    def test_cylinder_base(self):
        l=600
        r=400
        with self.assertRaises(ValueError):
            CylinderBase(l, r, -1.5, .5)
        with self.assertRaises(ValueError):
            CylinderBase(l, r, -.5, 1.5)
        with self.assertRaises(ValueError):
            CylinderBase(l,r,.5,-.5)
        c = CylinderBase(l, r, -1, 1)
        self.check_diff_etendue(c, l, r)
        with self.assertRaises(NotImplementedError):
            c.pdf(.5)
        self.assertEqual(c,c)

    def test_volume_corr(self):
        last_c1 = None
        for l in range(100, 1000, 300):
            for r in range(100, 1000, 300):
                c1 = VolumeCorrCylinder(l, r, -1, 1)
                self.check_diff_etendue(c1, l, r)
                self.check_pdf_etendue(c1, 2 * np.pi**2 * r * (r + l))

                c2 = VolumeCorrCylinder(l, r, -1, 0)
                self.check_pdf_etendue(c2, np.pi**2 * r * (r + l))

                c3 = VolumeCorrCylinder(l, r, 0, 1)
                self.check_pdf_etendue(c3, np.pi**2 * r * (r + l))

                self.assertEqual(c1,c1)
                self.assertNotEqual(c1,c2)
                self.assertNotEqual(c1,c3)
                self.assertEqual(c2,c2)
                self.assertNotEqual(c2,c3)
                self.assertEqual(c3,c3)
                self.assertNotEqual(c1,last_c1)
                last_c1 = c1

                s = str(c1)
                self.assertEqual(s[:19],'VolumeCorrCylinder(')
                x = s[19:-1].split(',')
                self.assertEqual(float(x[0]),l)
                self.assertEqual(float(x[1]),r)
                self.assertEqual(float(x[2]),-1)
                self.assertEqual(float(x[3]),1)

if __name__ == '__main__':
    unittest.main()

