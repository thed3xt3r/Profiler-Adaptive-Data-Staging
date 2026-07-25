import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import train


class ProfilerPolicyTests(unittest.TestCase):
    def test_classify_bottleneck_prefers_shard_for_decode(self):
        metrics = {
            "t_data": 0.08,
            "t_decode": 0.25,
            "t_h2d": 0.04,
            "t_gpu": 0.03,
        }
        self.assertEqual(train.classify_bottleneck(metrics), "shard")

    def test_select_policy_uses_stage_when_h2d_is_high(self):
        metrics = {
            "t_data": 0.05,
            "t_decode": 0.06,
            "t_h2d": 0.20,
            "t_gpu": 0.03,
        }
        policy = train.select_policy(metrics, scratch_available=True)
        self.assertEqual(policy, "stage")


if __name__ == "__main__":
    unittest.main()
