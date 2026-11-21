import unittest

from agent_core.plugins.builtins import register_builtin_distributions, builtin_distribution_count
from agent_core.registry import DistributionRegistry
from agent_core.service import SimulationService


class BuiltinDistributionTests(unittest.TestCase):
    def setUp(self):
        self.registry = DistributionRegistry()
        register_builtin_distributions(self.registry)
        self.service = SimulationService(self.registry, max_samples=100_000)

    def test_all_27_distributions_registered(self):
        self.assertEqual(builtin_distribution_count(), 27)
        self.assertEqual(len(self.registry), 27)

    def test_all_new_builtin_plugins_generate_requested_sample_count(self):
        cases = {
            "beta": {"alpha": 2.0, "beta": 3.0},
            "binomial": {"trials": 10, "p": 0.3},
            "cauchy": {"location": 0.0, "scale": 1.0},
            "chi_squared": {"degrees_of_freedom": 4.0},
            "exponential": {"mean": 2.0},
            "f_distribution": {"numerator_df": 5.0, "denominator_df": 10.0},
            "gamma": {"shape": 2.0, "scale": 3.0},
            "geometric": {"p": 0.3},
            "gumbel": {"location": 0.0, "scale": 1.0},
            "hypergeometric": {"population_size": 50, "number_of_successes": 20, "sample_size": 10},
            "kolmogorov_smirnov": {"sample_size": 20},
            "laplace": {"location": 0.0, "scale": 1.0},
            "levy": {"location": 0.0, "scale": 1.0},
            "logistic": {"location": 0.0, "scale": 1.0},
            "multivariate_normal": {"means": [0.0, 1.0], "covariances": [[1.0, 0.3], [0.3, 2.0]]},
            "nakagami": {"mu": 1.0, "omega": 2.0},
            "normal": {"mean": 0.0, "standard_deviation": 1.0},
            "pareto": {"scale": 1.0, "shape": 3.0},
            "pascal": {"successes": 4, "p": 0.4},
            "poisson": {"mean": 4.0},
            "student_t": {"degrees_of_freedom": 7.0},
            "triangular": {"lower": 0.0, "mode": 0.5, "upper": 1.0},
            "uniform_integer": {"lower": 1, "upper": 6},
            "uniform_real": {"lower": -2.0, "upper": 3.0},
            "weibull": {"scale": 2.0, "shape": 1.5},
            "zipf": {"number_of_elements": 20, "exponent": 1.2},
        }
        self.assertEqual(len(cases), 26)
        for distribution_id, params in cases.items():
            with self.subTest(distribution=distribution_id):
                result = self.service.simulate(distribution_id, params, 100, seed=2026)
                self.assertEqual(len(result.samples), 100)

    def test_seed_reproducibility_for_all_new_plugins(self):
        cases = {
            "beta": {"alpha": 2.0, "beta": 3.0},
            "binomial": {"trials": 5, "p": 0.3},
            "cauchy": {"location": 0.0, "scale": 1.0},
            "chi_squared": {"degrees_of_freedom": 4.0},
            "exponential": {"mean": 2.0},
            "f_distribution": {"numerator_df": 5.0, "denominator_df": 10.0},
            "gamma": {"shape": 2.0, "scale": 3.0},
            "geometric": {"p": 0.3},
            "gumbel": {"location": 0.0, "scale": 1.0},
            "hypergeometric": {"population_size": 50, "number_of_successes": 20, "sample_size": 10},
            "kolmogorov_smirnov": {"sample_size": 10},
            "laplace": {"location": 0.0, "scale": 1.0},
            "levy": {"location": 0.0, "scale": 1.0},
            "logistic": {"location": 0.0, "scale": 1.0},
            "multivariate_normal": {"means": [0.0, 1.0], "covariances": [[1.0, 0.3], [0.3, 2.0]]},
            "nakagami": {"mu": 1.0, "omega": 2.0},
            "normal": {"mean": 0.0, "standard_deviation": 1.0},
            "pareto": {"scale": 1.0, "shape": 3.0},
            "pascal": {"successes": 4, "p": 0.4},
            "poisson": {"mean": 4.0},
            "student_t": {"degrees_of_freedom": 7.0},
            "triangular": {"lower": 0.0, "mode": 0.5, "upper": 1.0},
            "uniform_integer": {"lower": 1, "upper": 6},
            "uniform_real": {"lower": -2.0, "upper": 3.0},
            "weibull": {"scale": 2.0, "shape": 1.5},
            "zipf": {"number_of_elements": 20, "exponent": 1.2},
        }
        for distribution_id, params in cases.items():
            with self.subTest(distribution=distribution_id):
                a = self.service.simulate(distribution_id, params, 20, seed=99)
                b = self.service.simulate(distribution_id, params, 20, seed=99)
                self.assertEqual(a.samples, b.samples)

    def test_supports_and_special_parameterizations(self):
        geom = self.service.simulate("geometric", {"p": 1.0}, 50, seed=1)
        self.assertEqual(set(geom.samples), {0})

        pascal = self.service.simulate("pascal", {"successes": 5, "p": 1.0}, 50, seed=1)
        self.assertEqual(set(pascal.samples), {0})

        zipf = self.service.simulate(
            "zipf", {"number_of_elements": 7, "exponent": 1.1}, 500, seed=1
        )
        self.assertTrue(all(1 <= x <= 7 for x in zipf.samples))

        hyper = self.service.simulate(
            "hypergeometric",
            {"population_size": 30, "number_of_successes": 12, "sample_size": 8},
            500, seed=1
        )
        self.assertTrue(all(0 <= x <= 8 for x in hyper.samples))

        ks = self.service.simulate("kolmogorov_smirnov", {"sample_size": 20}, 100, seed=1)
        self.assertTrue(all(0 <= x <= 1 for x in ks.samples))

        mvn = self.service.simulate(
            "multivariate_normal",
            {"means": [0.0, 1.0], "covariances": [[1.0, 0.2], [0.2, 1.0]]},
            50, seed=1
        )
        self.assertTrue(all(len(x) == 2 for x in mvn.samples))

    def test_logistic_sampler_uses_correct_inverse_transform(self):
        result = self.service.simulate(
            "logistic",
            {"location": 2.0, "scale": 1.5},
            30_000,
            seed=2026,
        )
        mean = sum(result.samples) / len(result.samples)
        self.assertAlmostEqual(mean, 2.0, delta=0.08)

    def test_rejects_invalid_covariance_matrix(self):
        with self.assertRaises(ValueError):
            self.service.simulate(
                "multivariate_normal",
                {"means": [0.0, 0.0], "covariances": [[1.0, 2.0], [2.0, 1.0]]},
                10, seed=1
            )

    def test_rejects_invalid_distribution_parameters(self):
        with self.assertRaises(ValueError):
            self.service.simulate("beta", {"alpha": 0, "beta": 2}, 10, seed=1)
        with self.assertRaises(ValueError):
            self.service.simulate("uniform_integer", {"lower": 5, "upper": 2}, 10, seed=1)
        with self.assertRaises(ValueError):
            self.service.simulate("nakagami", {"mu": 0.2, "omega": 1}, 10, seed=1)


if __name__ == "__main__":
    unittest.main()
