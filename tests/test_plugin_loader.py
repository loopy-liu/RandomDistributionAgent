import tempfile
import unittest
from pathlib import Path

from agent_core.plugin_loader import load_external_plugins
from agent_core.plugins.builtins import register_builtin_distributions
from agent_core.registry import DistributionRegistry
from agent_core.service import SimulationService


EXAMPLE_PLUGIN = Path(__file__).parents[1] / "examples" / "plugin_bernoulli.py"


class PluginLoaderTests(unittest.TestCase):
    def test_discovers_and_registers_trusted_plugin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "plugin_bernoulli.py"
            target.write_text(EXAMPLE_PLUGIN.read_text(encoding="utf-8"), encoding="utf-8")
            registry = DistributionRegistry()
            report = load_external_plugins(registry, temp_dir)
            self.assertEqual(report.loaded_ids, ("demo_bernoulli_extension",))
            self.assertEqual(report.errors, ())
            result = SimulationService(registry).simulate(
                "demo_bernoulli_extension", {"p": 0.7}, 100, seed=2026
            )
            self.assertEqual(len(result.samples), 100)
            self.assertTrue(set(result.samples).issubset({0, 1}))

    def test_does_not_load_files_without_plugin_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "not_a_plugin.py").write_text(
                "raise RuntimeError('should not execute')", encoding="utf-8"
            )
            report = load_external_plugins(DistributionRegistry(), temp_dir)
            self.assertEqual(report.loaded_ids, ())
            self.assertEqual(report.errors, ())

    def test_reports_missing_factory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "plugin_invalid.py").write_text(
                "VALUE = 1", encoding="utf-8"
            )
            report = load_external_plugins(DistributionRegistry(), temp_dir)
            self.assertEqual(report.loaded_ids, ())
            self.assertEqual(len(report.errors), 1)
            self.assertIn("build_plugin", report.errors[0])

    def test_duplicate_builtin_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = EXAMPLE_PLUGIN.read_text(encoding="utf-8").replace(
                "demo_bernoulli_extension", "normal"
            )
            (Path(temp_dir) / "plugin_duplicate.py").write_text(source, encoding="utf-8")
            registry = DistributionRegistry()
            register_builtin_distributions(registry)
            report = load_external_plugins(registry, temp_dir)
            self.assertEqual(report.loaded_ids, ())
            self.assertEqual(len(report.errors), 1)
            self.assertIn("已注册", report.errors[0])

    def test_reserved_custom_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = EXAMPLE_PLUGIN.read_text(encoding="utf-8").replace(
                "demo_bernoulli_extension", "custom_conflict"
            )
            (Path(temp_dir) / "plugin_reserved.py").write_text(source, encoding="utf-8")
            report = load_external_plugins(DistributionRegistry(), temp_dir)
            self.assertEqual(report.loaded_ids, ())
            self.assertEqual(len(report.errors), 1)
            self.assertIn("保留前缀", report.errors[0])


if __name__ == "__main__":
    unittest.main()
