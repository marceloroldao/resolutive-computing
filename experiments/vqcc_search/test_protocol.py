"""Budget-accounting checks; these do not certify a quantum circuit."""
import unittest
import numpy as np
from search import METHODS, INCUMBENT, LOW, HIGH, decode, optimize_one

class ProtocolTests(unittest.TestCase):
    def test_decode_bounds(self):
        self.assertEqual(decode(np.zeros(5)), tuple(LOW))
        self.assertEqual(decode(np.ones(5)), tuple(HIGH))
        self.assertEqual(decode(np.full(5, -3.0)), tuple(LOW))
        self.assertEqual(decode(np.full(5, 3.0)), tuple(HIGH))

    def test_equal_budget_and_shared_incumbent(self):
        for method in METHODS:
            with self.subTest(method=method):
                calls = []
                def objective(config):
                    calls.append(config)
                    return dict(configuration=list(config),
                                objective=float(sum((a-b)**2 for a,b in zip(config,INCUMBENT))))
                result = optimize_one(method, 0, objective, budget=1024)
                self.assertEqual(len(calls), 1024)
                self.assertEqual(result['calls'], 1024)
                self.assertEqual(calls[0], INCUMBENT)
                self.assertEqual(result['selected']['objective'], 0.0)

if __name__ == '__main__':
    unittest.main()
