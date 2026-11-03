"""CPU checks of the energy arithmetic; the samples are fixtures, never hardware measurements."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_probe as probe  # noqa: E402


class PowerProbeTests(unittest.TestCase):
    def test_energy_is_mean_power_times_period_after_settling(self):
        samples = [(t / 10, 10.0) for t in range(0, 200)] + [(20 + t / 10, 30.0) for t in range(0, 200)]
        windows = [(20 + i, 20 + i + 0.5) for i in range(20)]
        record = dict(method="m", inflight=2, prompt_tokens=1024, windows=windows, box_idle=dict(start=0, end=20),
                      loaded_idle=dict(start=0, end=20))
        row = probe.summarize_run(record, samples)
        self.assertEqual(row["prefills"], 15)  # windows that start in the first SETTLE_S seconds are dropped
        self.assertAlmostEqual(row["active_w"], 30.0)
        self.assertAlmostEqual(row["loaded_idle_w"], 10.0)
        self.assertAlmostEqual(row["period_ms"], (14.5 / 15) * 1000)
        self.assertAlmostEqual(row["prefill_median_ms"], 500.0)
        self.assertAlmostEqual(row["energy_j"], 30.0 * 14.5 / 15)
        self.assertAlmostEqual(row["net_of_loaded_idle_j"], 20.0 * 14.5 / 15)

    def test_update_period_counts_changes_of_the_reading(self):
        samples = [(i * 0.01, float(i // 10)) for i in range(100)]
        self.assertAlmostEqual(probe.update_period_ms(samples), 100.0)
        self.assertIsNone(probe.update_period_ms([(0.0, 5.0), (0.01, 5.0)]))


if __name__ == "__main__":
    unittest.main()
