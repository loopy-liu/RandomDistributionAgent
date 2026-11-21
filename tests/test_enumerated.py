import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from agent_core.registry import DistributionRegistry
from agent_core.service import SimulationService
from agent_core.plugins.enumerated import EnumeratedDistributionPlugin
from agent_core.xml_exporter import export_xml


class EnumeratedDistributionTests(unittest.TestCase):
    def setUp(self):
        self.registry = DistributionRegistry()
        self.registry.register(EnumeratedDistributionPlugin())
        self.service = SimulationService(self.registry)

    def test_sample_count_and_support(self):
        result = self.service.simulate(
            "enumerated",
            {"values": [1, 2, 3], "probabilities": [0.2, 0.3, 0.5]},
            sample_count=500,
            seed=2026,
        )
        self.assertEqual(len(result.samples), 500)
        self.assertTrue(set(result.samples).issubset({1, 2, 3}))

    def test_seed_reproducibility(self):
        params = {"values": ["A", "B"], "probabilities": [0.4, 0.6]}
        first = self.service.simulate("enumerated", params, 100, seed=123)
        second = self.service.simulate("enumerated", params, 100, seed=123)
        self.assertEqual(first.samples, second.samples)

    def test_empirical_frequency_is_reasonable(self):
        result = self.service.simulate(
            "enumerated",
            {"values": [0, 1, 2], "probabilities": [0.1, 0.2, 0.7]},
            sample_count=20_000,
            seed=2026,
        )
        observed = [result.samples.count(i) / len(result.samples) for i in (0, 1, 2)]
        expected = [0.1, 0.2, 0.7]
        for actual, target in zip(observed, expected):
            self.assertLess(abs(actual - target), 0.02)

    def test_invalid_probability_sum(self):
        with self.assertRaisesRegex(ValueError, "概率之和"):
            self.service.simulate(
                "enumerated",
                {"values": [1, 2], "probabilities": [0.2, 0.2]},
                10,
                seed=1,
            )

    def test_invalid_duplicate_values(self):
        with self.assertRaisesRegex(ValueError, "不能重复"):
            self.service.simulate(
                "enumerated",
                {"values": [1, 1], "probabilities": [0.5, 0.5]},
                10,
                seed=1,
            )

    def test_sample_count_validation(self):
        with self.assertRaises(ValueError):
            self.service.simulate(
                "enumerated",
                {"values": [1], "probabilities": [1.0]},
                0,
                seed=1,
            )

    def test_xml_export_is_parseable(self):
        result = self.service.simulate(
            "enumerated",
            {"values": ["晴", "雨"], "probabilities": [0.7, 0.3]},
            20,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = export_xml(result, Path(temp_dir) / "result.xml")
            root = ET.parse(path).getroot()
            self.assertEqual(root.tag, "simulation")
            self.assertEqual(root.find("distribution").get("id"), "enumerated")
            self.assertEqual(len(root.findall("./samples/sample")), 20)


if __name__ == "__main__":
    unittest.main()
