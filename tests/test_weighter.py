#!/usr/bin/env python
import unittest

from simweights.utils import Null
from simweights.weighter import MapWeighter, Weighter


class TestWeighter(unittest.TestCase):
    def test_null(self):
        n1 = Null()
        n2 = Null()
        self.assertEqual(n1, n2)
        self.assertNotEqual(n1, None)

        self.assertEqual(n1 + None, None)
        self.assertEqual(n1 + [], [])
        self.assertEqual(n1 + 5, 5)
        self.assertEqual(None + n1, None)
        self.assertEqual({} + n1, {})
        self.assertEqual(5 + n1, 5)

    def test_weighter(self):
        w = Weighter(None, None)
        with self.assertRaises(NotImplementedError):
            w._get_surface_params()
        with self.assertRaises(NotImplementedError):
            w._get_flux_params()
        with self.assertRaises(NotImplementedError):
            w._get_event_weight()
        with self.assertRaises(ValueError):
            w + object()
        with self.assertRaises(NotImplementedError):
            MapWeighter._get_surface_map(None)


if __name__ == "__main__":
    unittest.main()
