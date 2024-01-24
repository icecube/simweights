#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import uproot
from numpy.testing import assert_array_equal

from simweights import Hoerandel
from simweights._utils import (
    Column,
    Const,
    check_random_state,
    constcol,
    corsika_to_pdg,
    get_column,
    get_table,
    has_column,
    has_table,
)


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
        assert_array_equal(get_column(t1, "a"), 10 * [3])
        assert_array_equal(get_column(t1, "b"), 5 * [3] + 5 * [4])
        with self.assertRaises(AttributeError):
            get_column(t1, "c")

        self.assertEqual(has_column(t2, "a"), True)
        self.assertEqual(has_column(t2, "b"), True)
        self.assertEqual(has_column(t2, "c"), False)
        assert_array_equal(get_column(t2, "a"), 10 * [7])
        assert_array_equal(get_column(t2, "b"), range(10))
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

    def test_dists(self):
        p1 = Const(33)
        p2 = Const(44)
        self.assertEqual(p1.pdf(), 33)
        self.assertEqual(p2.pdf(), 44)
        self.assertEqual(p1, p1)
        self.assertEqual(p2, p2)
        self.assertNotEqual(p1, p2)
        self.assertEqual(str(p1), "Const(33)")
        self.assertEqual(str(p1), "Const(33)")

        p1 = Column("energy")
        p2 = Column("cos_zen")
        assert_array_equal(p1.pdf(np.arange(1000)), 1 / np.arange(1000))
        assert_array_equal(p2.pdf(1 / np.arange(10)), np.arange(10, dtype=float))
        self.assertEqual(p1, p1)
        self.assertEqual(p2, p2)
        self.assertNotEqual(p1, p2)
        self.assertEqual(str(p1), "Column('energy',)")
        self.assertEqual(str(p2), "Column('cos_zen',)")

    def test_uproot(self):
        with tempfile.TemporaryFile() as f:
            file = uproot.recreate(f)
            a1 = np.arange(1000)
            a2 = 2 * np.arange(1000)
            file["tree1"] = {"branch1": a1, "branch2": a2}

            file = uproot.open(f)
            t = get_table(file, "tree1")
            assert_array_equal(get_column(t, "branch1"), a1)
            assert_array_equal(get_column(t, "branch2"), a2)

    def test_check_random_state(self):
        self.assertIsInstance(check_random_state(None), np.random.Generator)
        self.assertIsInstance(check_random_state(3), np.random.Generator)
        self.assertIsInstance(check_random_state(np.int16(3)), np.random.Generator)
        r = np.random.RandomState()
        self.assertEqual(check_random_state(r), r)
        g = np.random.default_rng()
        self.assertEqual(check_random_state(g), g)
        with self.assertRaises(ValueError):
            check_random_state(object())
        with self.assertRaises(ValueError):
            check_random_state(33.3)

    def test_corsika_to_pdg(self):
        c = [14, 402, 703, 904, 1105, 1206, 1407, 1608, 1909, 2010, 2311, 2412, 2713, 2814]
        c += [3115, 3216, 3517, 4018, 3919, 4020, 4521, 4822, 5123, 5224, 5525, 5626]
        pdgid = [int(i) for i in Hoerandel.pdgids]
        assert_array_equal(corsika_to_pdg(c), pdgid)


if __name__ == "__main__":
    unittest.main()
