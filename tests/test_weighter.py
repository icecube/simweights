#!/usr/bin/env python
import unittest

from simweights.utils import Null
from simweights.weighter import Weighter


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
        w1 = Weighter([], Null(), {"event_weight": ("a", "b")})
        w2 = Weighter([], Null(), {"event_weight": ("a", "c")})
        with self.assertRaises(ValueError):
            w1 + w2


if __name__ == "__main__":
    unittest.main()
