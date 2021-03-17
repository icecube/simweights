#!/usr/bin/env python 
import unittest
import numpy as np
from  simweights import VolumeCorrCylinder,VolumeDetCylinder

class TestCylinder(unittest.TestCase):
    def check_cylinder(self,c,l,r):
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
            c._diff_etendue(-1.01)
        with self.assertRaises(AssertionError):
            c._diff_etendue( 1.01)

        self.assertAlmostEqual(c.etendue, np.pi**2*r*(r+l))
        

    def test_volume_corr(self):
        for l in range(100,1000,100):
            for r in range(100,1000,100):
                c = VolumeCorrCylinder(l,r,0,1)
                self.check_cylinder(c,l,r)
        
if __name__ == '__main__':
    unittest.main()

