import unittest

from simweights.utils import Null


class TestUtil(unittest.TestCase):
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
