import unittest

from linking.scoring_system import Score


class TestScoreClass(unittest.TestCase):

    def test_basics(self):
        score = Score()
        score.foo = 3
        score.bar = 2
        self.assertEqual(3, score.foo)
        self.assertEqual(5, score.total())

    def test_addition(self):
        score = Score()
        score.foo = 1
        score.foo += 2.1
        self.assertEqual(3.1, score.foo)
