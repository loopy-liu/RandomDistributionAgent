import tempfile
import unittest
from pathlib import Path

from agent_core.custom_distributions import (
    CustomDistributionManager,
    CustomDistributionRepository,
    SafeMathExpression,
    UnsafeExpressionError,
)
from agent_core.registry import DistributionRegistry
from agent_core.service import SimulationService
from agent_core.xml_exporter import to_xml_element


class CustomPmfTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = CustomDistributionRepository(Path(self.temp_dir.name) / "pmf.sqlite3")
        self.registry = DistributionRegistry()
        self.manager = CustomDistributionManager(self.repository, self.registry)
        self.service = SimulationService(self.registry)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_poisson_like_pmf_and_sample(self):
        definition = self.manager.create_discrete(
            name="自定义泊松型 PMF",
            expression="exp(-4)*4**x/factorial(x)",
            lower=0,
            upper=15,
            description="有限支持集上的泊松型权重",
        )
        result = self.service.simulate(definition.distribution_id, {}, 30_000, seed=2026)
        self.assertEqual(len(result.samples), 30_000)
        self.assertTrue(all(isinstance(value, int) and 0 <= value <= 15 for value in result.samples))
        mean = sum(result.samples) / len(result.samples)
        self.assertAlmostEqual(mean, 4.0, delta=0.08)

    def test_create_binomial_like_pmf(self):
        definition = self.manager.create_discrete(
            name="二项型 PMF",
            expression="comb(10,x)*0.3**x*0.7**(10-x)",
            lower=0,
            upper=10,
        )
        result = self.service.simulate(definition.distribution_id, {}, 20_000, seed=17)
        mean = sum(result.samples) / len(result.samples)
        self.assertAlmostEqual(mean, 3.0, delta=0.08)

    def test_pmf_seed_reproducibility(self):
        definition = self.manager.create_discrete(
            name="可复现 PMF", expression="x+1", lower=0, upper=5
        )
        first = self.service.simulate(definition.distribution_id, {}, 200, seed=99)
        second = self.service.simulate(definition.distribution_id, {}, 200, seed=99)
        self.assertEqual(first.samples, second.samples)

    def test_pmf_crud_and_persistence(self):
        definition = self.manager.create_discrete(
            name="待修改 PMF", expression="1", lower=0, upper=2
        )
        self.assertEqual(definition.kind, "discrete_pmf")
        updated = self.manager.update_discrete(
            definition.distribution_id,
            name="修改后的 PMF",
            expression="x+1",
            lower=1,
            upper=4,
        )
        self.assertEqual(updated.name, "修改后的 PMF")
        self.assertEqual(self.registry.get(definition.distribution_id).metadata.name, "修改后的 PMF")

        new_registry = DistributionRegistry()
        CustomDistributionManager(self.repository, new_registry)
        self.assertEqual(new_registry.get(definition.distribution_id).metadata.name, "修改后的 PMF")

        self.manager.delete(definition.distribution_id)
        self.assertEqual(len(self.repository.list_all()), 0)
        self.assertEqual(len(self.registry), 0)

    def test_pmf_definition_is_in_xml_parameters(self):
        definition = self.manager.create_discrete(
            name="XML PMF", expression="x+1", lower=0, upper=3
        )
        result = self.service.simulate(definition.distribution_id, {}, 12, seed=3)
        root = to_xml_element(result)
        params = {item.get("name"): item.text for item in root.findall("./distribution/parameters/parameter")}
        self.assertIn('"discrete_pmf"', params["kind"])
        self.assertIn("x+1", params["expression"])
        self.assertEqual(params["lower"], "0")
        self.assertEqual(params["upper"], "3")

    def test_reject_negative_or_zero_pmf(self):
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            self.manager.create_discrete(name="负权重", expression="x-2", lower=0, upper=2)
        with self.assertRaisesRegex(ValueError, "总和必须大于 0"):
            self.manager.create_discrete(name="零权重", expression="0", lower=0, upper=5)

    def test_reject_non_integer_support(self):
        with self.assertRaisesRegex(ValueError, "必须是整数"):
            self.manager.create_discrete(name="非整数支持集", expression="1", lower=0.5, upper=2)

    def test_safe_factorial_and_comb_are_bounded(self):
        self.assertEqual(SafeMathExpression("factorial(x)").evaluate(5), 120.0)
        self.assertEqual(SafeMathExpression("comb(5,x)").evaluate(2), 10.0)
        with self.assertRaises(ValueError):
            SafeMathExpression("factorial(x)").evaluate(171)
        with self.assertRaises(UnsafeExpressionError):
            SafeMathExpression("__import__('os')")

    def test_existing_pdf_definition_is_reloaded_and_exported(self):
        definition = self.manager.create(
            name="PDF 持久化兼容", expression="1", lower=0, upper=1
        )
        new_registry = DistributionRegistry()
        CustomDistributionManager(self.repository, new_registry)
        result = SimulationService(new_registry).simulate(
            definition.distribution_id, {}, 5, seed=8
        )
        root = to_xml_element(result)
        params = {item.get("name"): item.text for item in root.findall("./distribution/parameters/parameter")}
        self.assertIn('"continuous_pdf"', params["kind"])
        self.assertIn("expression", params)
        self.assertEqual(len(result.samples), 5)


if __name__ == "__main__":
    unittest.main()
