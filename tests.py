#!/usr/bin/env python3.11
import unittest
from tensor.tensor import Value


class TestValBasic(unittest.TestCase):
    def test_basic_ops(self):
        val1 = Value(2.0)
        val2 = Value(3.0)
        sq = (val2 - val1) ** 2
        self.assertEqual(sq._val, 1.0)
        sq.backward()
        self.assertEqual(val1._grad, -2.0)
        self.assertEqual(val2._grad, 2.0)


if __name__ == "__main__":
    unittest.main()
