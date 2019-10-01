import unittest
from fra_heur import *
import logging

class MyTestCase(unittest.TestCase):
    def test_something(self):
        S = get_switching_points([5,5.4],[8,2.6])
        S_exp = [1/6,9/28,1/2,19/28,5/6]
        for i, switch in enumerate(S):
            self.assertAlmostEqual(switch, S_exp[i])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
