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
        w = Weighter(None, None)
        with self.assertRaises(ValueError):
            w + object()


if __name__ == "__main__":
    unittest.main()
