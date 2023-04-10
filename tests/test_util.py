#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import unittest
from types import SimpleNamespace

import numpy as np
from simweights import Hoerandel
from simweights._utils import constcol, corsika_to_pdg, get_column, get_table, has_column, has_table


class TestUtil(unittest.TestCase):
    def test_table_and_column(self):
        t1 = SimpleNamespace(cols=SimpleNamespace(a=np.full(10, 3), b=np.array(5 * [3] + 5 * [4])))
        f1 = SimpleNamespace(root=SimpleNamespace(x=t1))
        t2 = {"a": np.full(10, 7), "b": np.arange(10)}
        f2 = {"x": t2}

        self.assertEqual(has_table(f1, "x"), True)
        self.assertEqual(has_table(f1, "y"), False)
        self.assertEqual(get_table(f1, "x"), t1)
        with self.assertRaises(AttributeError):
            get_table(f1, "y")

        self.assertEqual(has_table(f2, "x"), True)
        self.assertEqual(has_table(f2, "y"), False)
        self.assertEqual(get_table(f2, "x"), t2)
        with self.assertRaises(KeyError):
            get_table(f2, "y")

        self.assertEqual(has_column(t1, "a"), True)
        self.assertEqual(has_column(t1, "b"), True)
        self.assertEqual(has_column(t1, "c"), False)
        np.testing.assert_array_equal(get_column(t1, "a"), 10 * [3])
        np.testing.assert_array_equal(get_column(t1, "b"), 5 * [3] + 5 * [4])
        with self.assertRaises(AttributeError):
            get_column(t1, "c")

        self.assertEqual(has_column(t2, "a"), True)
        self.assertEqual(has_column(t2, "b"), True)
        self.assertEqual(has_column(t2, "c"), False)
        np.testing.assert_array_equal(get_column(t2, "a"), 10 * [7])
        np.testing.assert_array_equal(get_column(t2, "b"), range(10))
        with self.assertRaises(KeyError):
            get_column(t2, "c")

        mask = np.arange(10) < 5
        self.assertEqual(constcol(t1, "a"), 3)
        self.assertEqual(constcol(t1, "b", mask), 3)
        self.assertEqual(constcol(t1, "b", ~mask), 4)
        with self.assertRaises(AssertionError):
            constcol(t1, "b")
        with self.assertRaises(AttributeError):
            constcol(t1, "c")
        self.assertEqual(constcol(t2, "a"), 7)
        with self.assertRaises(AssertionError):
            constcol(t2, "b", mask)
        with self.assertRaises(AssertionError):
            constcol(t2, "b", ~mask)
        with self.assertRaises(AssertionError):
            constcol(t2, "b")
        with self.assertRaises(KeyError):
            constcol(t2, "c")

    def test_corsika_to_pdg(self):
        c = [14, 402, 703, 904, 1105, 1206, 1407, 1608, 1909, 2010, 2311, 2412, 2713, 2814]
        c += [3115, 3216, 3517, 4018, 3919, 4020, 4521, 4822, 5123, 5224, 5525, 5626]
        pdgid = [int(i) for i in Hoerandel.pdgids]
        np.testing.assert_array_equal(corsika_to_pdg(c), pdgid)


if __name__ == "__main__":
    unittest.main()
