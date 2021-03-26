#!/usr/bin/env python
import os, unittest, h5py, tables, pandas
import numpy as np
import simweights

weight_dtype = [('type',np.int32),
                ('energy',np.float64),
                ('zenith',np.float64),
                ('weight',np.float64),]
info_dtype = [('primary_type',np.int32),
              ('n_events',np.int32),
              ('cylinder_height',np.float64),
              ('cylinder_radius',np.float64),
              ('min_zenith',np.float64),
              ('max_zenith',np.float64),
              ('min_energy',np.float64),
              ('max_energy',np.float64),              
              ('power_law_index',np.float64)             
              ]

def make_hdf5_file(fname,v):
    weight = np.zeros(v[1], dtype=weight_dtype)
    weight['type'] = v[0]
    weight['zenith'] = np.linspace(v[4], v[5], v[1])
    weight['weight'] = 1
    if v[8] == -1:
        weight['energy'] = np.geomspace(v[6],v[7],v[1])
    else:
        q = np.linspace(1/2/v[1],1-1/2/v[1],v[1])
        G = v[8] + 1
        weight['energy'] = (q * (v[7]**G - v[6]**G) + v[6]**G)**(1 /G)
    info = np.array([v], dtype=info_dtype)
    f = h5py.File(fname,'w')
    f.create_dataset("I3CorsikaWeight", data = weight)
    f.create_dataset("I3PrimaryInjectorInfo", data = info)
    f.close()

class TestCylinder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        make_hdf5_file('file1.h5', (2212, 100000, 1200, 600,0, np.pi, 1e4, 1e6, -1))
        make_hdf5_file('file2.h5', (2212, 100000, 1200, 600,0, np.pi, 1e5, 1e7, -1.5))
        cls.etendue = simweights.VolumeCorrCylinder(600, 1200, 0, 1).etendue
        cls.flux_model = lambda cls, E, ptype: 1 / cls.etendue

    @classmethod
    def tearDownClass(cls):
        os.unlink('file1.h5')
        os.unlink('file2.h5')

    def check_weights(self, wf):
        w = wf.get_weights(self.flux_model)
        emin, emax = wf.surface.get_energy_range(2212)
        self.assertAlmostEqual(w.sum() / (emax - emin), 1, 4)
        E = wf.event_data["energy"]
        y,x = np.histogram(E, weights = w, bins=50, range=[emin, emax])
        Ewidth = (x[1:]-x[:-1])
        np.testing.assert_array_almost_equal(y/Ewidth, 1,2)

    def test_h5py(self):
        simfile = h5py.File('file1.h5','r')
        wf = simweights.PrimaryWeighter(simfile)
        self.check_weights(wf)

    def test_pytables(self):
        simfile = tables.open_file('file2.h5','r')
        wf = simweights.PrimaryWeighter(simfile)
        self.check_weights(wf)        

    def test_pandas(self):
        simfile = pandas.HDFStore('file1.h5','r')
        wf = simweights.PrimaryWeighter(simfile)
        self.check_weights(wf)

    def test_addition(self):
        simfile1 = h5py.File('file1.h5','r')
        simfile2 = pandas.HDFStore('file2.h5','r')
        wf1 = simweights.PrimaryWeighter(simfile1)
        wf2 = simweights.PrimaryWeighter(simfile2)
        wf = wf1 + wf2
        self.check_weights(wf)                

    def test_no_info(self):
        f = h5py.File('nothing.h5','w')
        f.create_dataset("I3CorsikaWeight", data = [])
        f.close()
        with self.assertRaises(RuntimeError):
            simfile1 = h5py.File('nothing.h5','r')
            wf1 = simweights.PrimaryWeighter(simfile1)
        os.unlink('nothing.h5')

if __name__ == '__main__':
    unittest.main()

