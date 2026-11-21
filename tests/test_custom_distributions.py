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


class CustomDistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database = Path(self.temp_dir.name) / "custom.sqlite3"
        self.repository = CustomDistributionRepository(database)
        self.registry = DistributionRegistry()
        self.manager = CustomDistributionManager(self.repository, self.registry)
        self.service = SimulationService(self.registry)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_and_sample_uniform_pdf(self):
        definition = self.manager.create(
            name="自定义均匀分布",
            expression="1",
            lower=0,
            upper=1,
            grid_size=2001,
            description="测试用常数密度",
        )
        result = self.service.simulate(definition.distribution_id, {}, 10000, seed=2026)
        self.assertEqual(len(result.samples), 10000)
        self.assertTrue(all(0 <= x <= 1 for x in result.samples))
        mean = sum(result.samples) / len(result.samples)
        self.assertAlmostEqual(mean, 0.5, delta=0.02)

    def test_create_and_sample_triangular_like_pdf(self):
        definition = self.manager.create(
            name="递增密度示例",
            expression="2*x",
            lower=0,
            upper=1,
            grid_size=4001,
        )
        result = self.service.simulate(definition.distribution_id, {}, 12000, seed=7)
        mean = sum(result.samples) / len(result.samples)
        # f(x)=2x on [0,1] has theoretical mean 2/3.
        self.assertAlmostEqual(mean, 2/3, delta=0.02)

    def test_crud_and_registry_refresh(self):
        definition = self.manager.create(
            name="待修改分布", expression="1", lower=0, upper=1
        )
        self.assertEqual(len(self.repository.list_all()), 1)
        self.assertEqual(len(self.registry), 1)

        updated = self.manager.update(
            definition.distribution_id,
            name="修改后的分布",
            expression="x + 1",
            lower=0,
            upper=1,
        )
        self.assertEqual(updated.name, "修改后的分布")
        self.assertEqual(self.registry.get(definition.distribution_id).metadata.name,
                         "修改后的分布")

        self.manager.delete(definition.distribution_id)
        self.assertEqual(self.repository.list_all(), [])
        self.assertEqual(len(self.registry), 0)

    def test_definition_survives_manager_restart(self):
        definition = self.manager.create(
            name="持久化分布", expression="1", lower=-1, upper=1
        )
        new_registry = DistributionRegistry()
        CustomDistributionManager(self.repository, new_registry)
        self.assertEqual(
            new_registry.get(definition.distribution_id).metadata.name,
            "持久化分布",
        )

    def test_reject_negative_density(self):
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            self.manager.create(
                name="非法密度", expression="x", lower=-1, upper=0
            )

    def test_reject_zero_integral(self):
        with self.assertRaisesRegex(ValueError, "积分必须大于 0"):
            self.manager.create(
                name="零密度", expression="0", lower=0, upper=1
            )

    def test_reject_unsafe_python(self):
        with self.assertRaises(UnsafeExpressionError):
            SafeMathExpression("__import__('os').system('echo unsafe')")

    def test_reject_unknown_name(self):
        with self.assertRaises(UnsafeExpressionError):
            SafeMathExpression("secret_value + x")


if __name__ == "__main__":
    unittest.main()
