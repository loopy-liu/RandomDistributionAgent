from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Callable

from ..contracts import (
    DistributionMetadata,
    ParameterDefinition,
    RandomSource,
    SampleValue,
)
from ..registry import DistributionRegistry
from .enumerated import EnumeratedDistributionPlugin


def _u(rng: RandomSource) -> float:
    # Keep inverse transforms away from exact 0 and 1.
    return min(1.0 - 1e-15, max(1e-15, rng.random()))


def _normal(rng: RandomSource) -> float:
    u1 = _u(rng)
    u2 = _u(rng)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _gamma(rng: RandomSource, shape: float, scale: float = 1.0) -> float:
    """Marsaglia-Tsang gamma sampler; shape and scale use the usual convention."""
    if shape <= 0 or scale <= 0:
        raise ValueError("Gamma 的 shape 和 scale 必须大于 0")
    if shape < 1.0:
        return _gamma(rng, shape + 1.0, scale) * (_u(rng) ** (1.0 / shape))

    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = _normal(rng)
        one_plus = 1.0 + c * x
        if one_plus <= 0:
            continue
        v = one_plus ** 3
        u = _u(rng)
        if u < 1.0 - 0.0331 * (x ** 4):
            return scale * d * v
        if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
            return scale * d * v


def _geometric_failures(rng: RandomSource, p: float) -> int:
    if p >= 1.0:
        return 0
    return int(math.floor(math.log1p(-_u(rng)) / math.log1p(-p)))


def _poisson_knuth(rng: RandomSource, lam: float) -> int:
    if lam == 0:
        return 0
    threshold = math.exp(-lam)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= _u(rng)
    return count - 1


def _poisson_exact(rng: RandomSource, lam: float) -> int:
    # Split a large Poisson mean into independent smaller Poisson variables.
    # The sum remains exactly Poisson(lambda), avoiding a normal approximation.
    if lam == 0:
        return 0
    chunks = max(1, math.ceil(lam / 20.0))
    part = lam / chunks
    return sum(_poisson_knuth(rng, part) for _ in range(chunks))


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            value = matrix[i][j] - sum(
                lower[i][k] * lower[j][k] for k in range(j)
            )
            if i == j:
                if value <= 1e-12:
                    raise ValueError(
                        "协方差矩阵必须对称正定（当前版本暂不支持奇异半正定矩阵）"
                    )
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    return lower


def _number(params: dict[str, Any], key: str, *, minimum=None, maximum=None) -> float:
    value = params.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"参数 {key} 必须是数字")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"参数 {key} 必须是有限数字")
    if minimum is not None and value < minimum:
        raise ValueError(f"参数 {key} 必须大于等于 {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"参数 {key} 必须小于等于 {maximum}")
    return value


def _integer(params: dict[str, Any], key: str, *, minimum=None, maximum=None) -> int:
    value = params.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"参数 {key} 必须是整数")
    if minimum is not None and value < minimum:
        raise ValueError(f"参数 {key} 必须大于等于 {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"参数 {key} 必须小于等于 {maximum}")
    return value


def _probability(params: dict[str, Any], key: str, *, allow_zero=False) -> float:
    value = _number(params, key, minimum=0.0 if allow_zero else 1e-15, maximum=1.0)
    return value


def _positive(params: dict[str, Any], key: str) -> float:
    return _number(params, key, minimum=1e-15)


def _validate_exact(params: dict[str, Any], keys: set[str]) -> None:
    missing = keys - set(params)
    extra = set(params) - keys
    if missing:
        raise ValueError(f"缺少参数：{', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"存在不支持的参数：{', '.join(sorted(extra))}")


def _workload(sample_count: int, cost_per_sample: int, label: str) -> None:
    if sample_count * cost_per_sample > 20_000_000:
        raise ValueError(
            f"{label} 的计算量过大。请减少样本数量或降低相关参数，"
            "单次计算工作量上限为 2,000 万次基本循环。"
        )


@dataclass
class CallableDistributionPlugin:
    _metadata: DistributionMetadata
    _validator: Callable[[dict[str, Any]], dict[str, Any]]
    _sampler: Callable[[int, dict[str, Any], RandomSource], list[SampleValue]]

    @property
    def metadata(self) -> DistributionMetadata:
        return self._metadata

    def validate_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._validator(params)

    def sample(
        self,
        n: int,
        params: dict[str, Any],
        rng: RandomSource,
    ) -> list[SampleValue]:
        return self._sampler(n, params, rng)


def _param(name, label, value_type, description, default, minimum=None, maximum=None):
    return ParameterDefinition(
        name=name,
        label=label,
        value_type=value_type,
        description=description,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def _plugin(
    distribution_id: str,
    name: str,
    kind: str,
    params: tuple[ParameterDefinition, ...],
    validator,
    sampler,
    description: str,
    dimension: str = "1D",
) -> CallableDistributionPlugin:
    return CallableDistributionPlugin(
        DistributionMetadata(
            distribution_id=distribution_id,
            name=name,
            kind=kind,
            dimension=dimension,
            version="1.0.0",
            parameters=params,
            description=description,
        ),
        validator,
        sampler,
    )


def _make_plugins() -> list[CallableDistributionPlugin]:
    plugins = []

    # 1. Beta(alpha, beta)
    def v_beta(p):
        _validate_exact(p, {"alpha", "beta"})
        return {"alpha": _positive(p, "alpha"), "beta": _positive(p, "beta")}
    plugins.append(_plugin(
        "beta", "Beta 分布", "continuous",
        (_param("alpha", "形状参数 α", "float", "必须大于 0", 2.0, 1e-15),
         _param("beta", "形状参数 β", "float", "必须大于 0", 5.0, 1e-15)),
        v_beta,
        lambda n, p, r: [
            (lambda a, b: a / (a + b))(
                _gamma(r, p["alpha"]), _gamma(r, p["beta"])
            ) for _ in range(n)
        ],
        "取值位于 [0,1] 的连续分布。"
    ))

    # 2. Binomial(trials, p)
    def v_binomial(p):
        _validate_exact(p, {"trials", "p"})
        return {
            "trials": _integer(p, "trials", minimum=0, maximum=1_000_000),
            "p": _probability(p, "p", allow_zero=True),
        }
    def s_binomial(n, p, r):
        trials, prob = p["trials"], p["p"]
        _workload(n, trials, "二项分布")
        if prob == 0:
            return [0] * n
        if prob == 1:
            return [trials] * n
        return [sum(1 for _ in range(trials) if r.random() < prob) for _ in range(n)]
    plugins.append(_plugin(
        "binomial", "二项分布", "discrete",
        (_param("trials", "试验次数 n", "int", "非负整数", 10, 0, 1_000_000),
         _param("p", "单次成功概率 p", "float", "取值范围 [0,1]", 0.5, 0, 1)),
        v_binomial, s_binomial,
        "固定次数独立伯努利试验中成功次数的分布。"
    ))

    # 3. Cauchy(location, scale)
    def v_cauchy(p):
        _validate_exact(p, {"location", "scale"})
        return {"location": _number(p, "location"), "scale": _positive(p, "scale")}
    plugins.append(_plugin(
        "cauchy", "柯西分布", "continuous",
        (_param("location", "位置参数 x₀", "float", "分布中心", 0.0),
         _param("scale", "尺度参数 γ", "float", "必须大于 0", 1.0, 1e-15)),
        v_cauchy,
        lambda n, p, r: [
            p["location"] + p["scale"] * math.tan(math.pi * (_u(r) - 0.5))
            for _ in range(n)
        ],
        "位置-尺度参数化的柯西分布。"
    ))

    # 4. Chi-square(df)
    def v_chi(p):
        _validate_exact(p, {"degrees_of_freedom"})
        return {"degrees_of_freedom": _positive(p, "degrees_of_freedom")}
    plugins.append(_plugin(
        "chi_squared", "卡方分布", "continuous",
        (_param("degrees_of_freedom", "自由度 ν", "float", "必须大于 0", 4.0, 1e-15),),
        v_chi,
        lambda n, p, r: [
            _gamma(r, p["degrees_of_freedom"] / 2.0, 2.0) for _ in range(n)
        ],
        "自由度为 ν 的卡方分布。"
    ))

    # 7. Exponential(mean)
    def v_exp(p):
        _validate_exact(p, {"mean"})
        return {"mean": _positive(p, "mean")}
    plugins.append(_plugin(
        "exponential", "指数分布", "continuous",
        (_param("mean", "均值参数 μ", "float", "均值必须大于 0", 1.0, 1e-15),),
        v_exp,
        lambda n, p, r: [-p["mean"] * math.log1p(-_u(r)) for _ in range(n)],
        "以均值 mean 参数化的指数分布。"
    ))

    # 8. F(df1, df2)
    def v_f(p):
        _validate_exact(p, {"numerator_df", "denominator_df"})
        return {
            "numerator_df": _positive(p, "numerator_df"),
            "denominator_df": _positive(p, "denominator_df"),
        }
    def s_f(n, p, r):
        a, b = p["numerator_df"], p["denominator_df"]
        return [
            (_gamma(r, a / 2.0, 2.0) / a) /
            (_gamma(r, b / 2.0, 2.0) / b)
            for _ in range(n)
        ]
    plugins.append(_plugin(
        "f_distribution", "F 分布", "continuous",
        (_param("numerator_df", "分子自由度 ν₁", "float", "必须大于 0", 5.0, 1e-15),
         _param("denominator_df", "分母自由度 ν₂", "float", "必须大于 0", 10.0, 1e-15)),
        v_f, s_f, "两个独立卡方变量分别除以自由度后的比值分布。"
    ))

    # 9. Gamma(shape, scale)
    def v_gamma(p):
        _validate_exact(p, {"shape", "scale"})
        return {"shape": _positive(p, "shape"), "scale": _positive(p, "scale")}
    plugins.append(_plugin(
        "gamma", "Gamma 分布", "continuous",
        (_param("shape", "形状参数 k", "float", "必须大于 0", 2.0, 1e-15),
         _param("scale", "尺度参数 θ", "float", "必须大于 0", 2.0, 1e-15)),
        v_gamma,
        lambda n, p, r: [_gamma(r, p["shape"], p["scale"]) for _ in range(n)],
        "采用 shape-scale 参数化的 Gamma 分布。"
    ))

    # 10. Geometric(p): failures before first success, support {0,1,...}
    def v_geom(p):
        _validate_exact(p, {"p"})
        return {"p": _probability(p, "p")}
    plugins.append(_plugin(
        "geometric", "几何分布", "discrete",
        (_param("p", "成功概率 p", "float", "取值范围 (0,1]", 0.3, 1e-15, 1),),
        v_geom,
        lambda n, p, r: [_geometric_failures(r, p["p"]) for _ in range(n)],
        "采用 Apache Commons Math 约定：记录首次成功前的失败次数，支持集为 0,1,2,...。"
    ))

    # 11. Gumbel(location, scale)
    def v_gumbel(p):
        _validate_exact(p, {"location", "scale"})
        return {"location": _number(p, "location"), "scale": _positive(p, "scale")}
    plugins.append(_plugin(
        "gumbel", "Gumbel 分布", "continuous",
        (_param("location", "位置参数 μ", "float", "位置参数", 0.0),
         _param("scale", "尺度参数 β", "float", "必须大于 0", 1.0, 1e-15)),
        v_gumbel,
        lambda n, p, r: [
            p["location"] - p["scale"] * math.log(-math.log(_u(r)))
            for _ in range(n)
        ],
        "采用最大值型 Gumbel 参数化。"
    ))

    # 12. Hypergeometric(N, K, sample_size)
    def v_hyper(p):
        _validate_exact(p, {"population_size", "number_of_successes", "sample_size"})
        N = _integer(p, "population_size", minimum=1, maximum=1_000_000)
        K = _integer(p, "number_of_successes", minimum=0, maximum=N)
        draws = _integer(p, "sample_size", minimum=0, maximum=N)
        return {"population_size": N, "number_of_successes": K, "sample_size": draws}
    def s_hyper(n, p, r):
        N, K, draws = p["population_size"], p["number_of_successes"], p["sample_size"]
        cost = min(draws, N - draws)
        _workload(n, cost, "超几何分布")
        output = []
        for _ in range(n):
            if draws > N // 2:
                complement_draws = N - draws
                succ = K
                fail = N - K
                got_complement = 0
                for _j in range(complement_draws):
                    if r.random() < succ / (succ + fail):
                        got_complement += 1
                        succ -= 1
                    else:
                        fail -= 1
                output.append(K - got_complement)
            else:
                succ = K
                fail = N - K
                got = 0
                for _j in range(draws):
                    if r.random() < succ / (succ + fail):
                        got += 1
                        succ -= 1
                    else:
                        fail -= 1
                output.append(got)
        return output
    plugins.append(_plugin(
        "hypergeometric", "超几何分布", "discrete",
        (_param("population_size", "总体容量 N", "int", "正整数", 50, 1, 1_000_000),
         _param("number_of_successes", "总体成功数 K", "int", "0 到 N 之间的整数", 20, 0, 1_000_000),
         _param("sample_size", "抽取样本量 n", "int", "0 到 N 之间的整数", 10, 0, 1_000_000)),
        v_hyper, s_hyper,
        "从含 K 个成功元素和 N-K 个失败元素的总体中无放回抽取 n 个元素，输出成功元素数。"
    ))

    # 13. Kolmogorov-Smirnov statistic D_n for Uniform(0,1)
    def v_ks(p):
        _validate_exact(p, {"sample_size"})
        return {"sample_size": _integer(p, "sample_size", minimum=1, maximum=100_000)}
    def s_ks(n, p, r):
        size = p["sample_size"]
        _workload(n, size, "Kolmogorov-Smirnov 分布")
        values = []
        for _ in range(n):
            ordered = sorted(_u(r) for _j in range(size))
            d_plus = max((i + 1) / size - x for i, x in enumerate(ordered))
            d_minus = max(x - i / size for i, x in enumerate(ordered))
            values.append(max(d_plus, d_minus))
        return values
    plugins.append(_plugin(
        "kolmogorov_smirnov", "Kolmogorov–Smirnov 分布", "continuous",
        (_param("sample_size", "样本量 n", "int", "构造 KS 统计量时使用的样本量", 20, 1, 100_000),),
        v_ks, s_ks,
        "通过重复抽取 U(0,1) 样本并计算双侧单样本 KS 统计量 Dₙ 来生成其分布。"
    ))

    # 14. Laplace(location, scale)
    def v_laplace(p):
        _validate_exact(p, {"location", "scale"})
        return {"location": _number(p, "location"), "scale": _positive(p, "scale")}
    def s_laplace(n, p, r):
        result = []
        for _ in range(n):
            u = _u(r)
            if u < 0.5:
                result.append(p["location"] + p["scale"] * math.log(2.0 * u))
            else:
                result.append(p["location"] - p["scale"] * math.log(2.0 * (1.0 - u)))
        return result
    plugins.append(_plugin(
        "laplace", "Laplace 分布", "continuous",
        (_param("location", "位置参数 μ", "float", "位置参数", 0.0),
         _param("scale", "尺度参数 b", "float", "必须大于 0", 1.0, 1e-15)),
        v_laplace, s_laplace, "位置-尺度参数化的双指数分布。"
    ))

    # 15. Levy(location, scale c)
    def v_levy(p):
        _validate_exact(p, {"location", "scale"})
        return {"location": _number(p, "location"), "scale": _positive(p, "scale")}
    plugins.append(_plugin(
        "levy", "Lévy 分布", "continuous",
        (_param("location", "位置参数 μ", "float", "位置参数", 0.0),
         _param("scale", "尺度参数 c", "float", "必须大于 0", 1.0, 1e-15)),
        v_levy,
        lambda n, p, r: [
            p["location"] + p["scale"] / (_normal(r) ** 2)
            for _ in range(n)
        ],
        "采用 Apache Commons Math 的位置参数 μ 和尺度参数 c；均值与方差均不有限。"
    ))

    # 16. Logistic(location, scale)
    def v_logistic(p):
        _validate_exact(p, {"location", "scale"})
        return {"location": _number(p, "location"), "scale": _positive(p, "scale")}
    plugins.append(_plugin(
        "logistic", "Logistic 分布", "continuous",
        (_param("location", "位置参数 μ", "float", "位置参数", 0.0),
         _param("scale", "尺度参数 s", "float", "必须大于 0", 1.0, 1e-15)),
        v_logistic,
        lambda n, p, r: [
            p["location"] + p["scale"] * math.log(u / (1.0 - u))
            for u in (_u(r) for _ in range(n))
        ],
        "采用位置-尺度参数化的 Logistic 分布。"
    ))

    # 17. Multivariate normal
    def v_mvn(p):
        _validate_exact(p, {"means", "covariances"})
        means = p.get("means")
        cov = p.get("covariances")
        if not isinstance(means, list) or not means:
            raise ValueError("means 必须是非空数值列表")
        if len(means) > 20:
            raise ValueError("当前版本多元正态最多支持 20 维")
        normalized_means = []
        for value in means:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("means 中的每一项必须是有限数字")
            normalized_means.append(float(value))
        dim = len(normalized_means)
        if not isinstance(cov, list) or len(cov) != dim:
            raise ValueError("covariances 必须是与均值向量维度一致的方阵")
        matrix = []
        for row in cov:
            if not isinstance(row, list) or len(row) != dim:
                raise ValueError("covariances 必须是方阵")
            normalized_row = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError("协方差矩阵元素必须是有限数字")
                normalized_row.append(float(value))
            matrix.append(normalized_row)
        for i in range(dim):
            for j in range(dim):
                if abs(matrix[i][j] - matrix[j][i]) > 1e-9:
                    raise ValueError("协方差矩阵必须对称")
        _cholesky(matrix)
        return {"means": normalized_means, "covariances": matrix}
    def s_mvn(n, p, r):
        means = p["means"]
        lower = _cholesky(p["covariances"])
        dim = len(means)
        result = []
        for _ in range(n):
            z = [_normal(r) for _j in range(dim)]
            vector = []
            for i in range(dim):
                vector.append(means[i] + sum(lower[i][j] * z[j] for j in range(i + 1)))
            result.append(vector)
        return result
    plugins.append(_plugin(
        "multivariate_normal", "多元正态分布", "continuous",
        (_param("means", "均值向量 μ", "vector_float", "例如 [0, 0]", [0.0, 0.0]),
         _param("covariances", "协方差矩阵 Σ", "matrix_float", "例如 [[1,0.5],[0.5,1]]", [[1.0, 0.5], [0.5, 1.0]])),
        v_mvn, s_mvn,
        "由均值向量和协方差矩阵定义；当前版本要求协方差矩阵对称正定。",
        dimension="多维"
    ))

    # 18. Nakagami(mu, omega)
    def v_nakagami(p):
        _validate_exact(p, {"mu", "omega"})
        mu = _number(p, "mu", minimum=0.5)
        omega = _positive(p, "omega")
        return {"mu": mu, "omega": omega}
    plugins.append(_plugin(
        "nakagami", "Nakagami 分布", "continuous",
        (_param("mu", "形状参数 μ", "float", "必须大于等于 0.5", 1.0, 0.5),
         _param("omega", "尺度参数 Ω", "float", "必须大于 0", 1.0, 1e-15)),
        v_nakagami,
        lambda n, p, r: [
            math.sqrt(_gamma(r, p["mu"], p["omega"] / p["mu"]))
            for _ in range(n)
        ],
        "参数约定与 Apache Commons Math 一致：μ≥0.5，Ω>0。"
    ))

    # 19. Normal(mean, standard deviation)
    def v_normal(p):
        _validate_exact(p, {"mean", "standard_deviation"})
        return {
            "mean": _number(p, "mean"),
            "standard_deviation": _positive(p, "standard_deviation"),
        }
    plugins.append(_plugin(
        "normal", "正态分布", "continuous",
        (_param("mean", "均值 μ", "float", "位置参数", 0.0),
         _param("standard_deviation", "标准差 σ", "float", "必须大于 0", 1.0, 1e-15)),
        v_normal,
        lambda n, p, r: [
            p["mean"] + p["standard_deviation"] * _normal(r)
            for _ in range(n)
        ],
        "采用均值 μ 和标准差 σ 参数化的正态分布。"
    ))

    # 20. Pareto(scale, shape)
    def v_pareto(p):
        _validate_exact(p, {"scale", "shape"})
        return {"scale": _positive(p, "scale"), "shape": _positive(p, "shape")}
    plugins.append(_plugin(
        "pareto", "Pareto 分布", "continuous",
        (_param("scale", "尺度参数 xₘ", "float", "最小取值，必须大于 0", 1.0, 1e-15),
         _param("shape", "形状参数 α", "float", "必须大于 0", 3.0, 1e-15)),
        v_pareto,
        lambda n, p, r: [
            p["scale"] * ((1.0 - _u(r)) ** (-1.0 / p["shape"]))
            for _ in range(n)
        ],
        "采用 Pareto I 分布，支持集为 [scale, +∞)。"
    ))

    # 21. Pascal(r,p), failures before r successes
    def v_pascal(p):
        _validate_exact(p, {"successes", "p"})
        return {
            "successes": _integer(p, "successes", minimum=1, maximum=1_000_000),
            "p": _probability(p, "p"),
        }
    def s_pascal(n, p, r):
        successes, prob = p["successes"], p["p"]
        _workload(n, successes, "Pascal 分布")
        if prob == 1.0:
            return [0] * n
        return [
            sum(_geometric_failures(r, prob) for _j in range(successes))
            for _ in range(n)
        ]
    plugins.append(_plugin(
        "pascal", "Pascal 分布", "discrete",
        (_param("successes", "目标成功次数 r", "int", "正整数", 5, 1, 1_000_000),
         _param("p", "单次成功概率 p", "float", "取值范围 (0,1]", 0.4, 1e-15, 1)),
        v_pascal, s_pascal,
        "采用 Apache Commons Math 约定：输出达到 r 次成功前的失败总次数。"
    ))

    # 22. Poisson(mean)
    def v_poisson(p):
        _validate_exact(p, {"mean"})
        return {"mean": _number(p, "mean", minimum=0.0, maximum=1_000_000.0)}
    def s_poisson(n, p, r):
        lam = p["mean"]
        _workload(n, max(1, math.ceil(lam / 20.0)) * max(1, int(20 + lam / max(1, math.ceil(lam / 20.0)))), "Poisson 分布")
        return [_poisson_exact(r, lam) for _ in range(n)]
    plugins.append(_plugin(
        "poisson", "Poisson 分布", "discrete",
        (_param("mean", "均值参数 λ", "float", "必须大于等于 0", 4.0, 0, 1_000_000),),
        v_poisson, s_poisson,
        "计数型分布，均值和方差均为 λ。"
    ))

    # 23. Student t(df)
    def v_t(p):
        _validate_exact(p, {"degrees_of_freedom"})
        return {"degrees_of_freedom": _positive(p, "degrees_of_freedom")}
    plugins.append(_plugin(
        "student_t", "Student t 分布", "continuous",
        (_param("degrees_of_freedom", "自由度 ν", "float", "必须大于 0", 5.0, 1e-15),),
        v_t,
        lambda n, p, r: [
            _normal(r) / math.sqrt(
                _gamma(r, p["degrees_of_freedom"] / 2.0, 2.0)
                / p["degrees_of_freedom"]
            )
            for _ in range(n)
        ],
        "由标准正态变量与独立卡方变量之比构造的 Student t 分布。"
    ))

    # 24. Triangular(lower, mode, upper)
    def v_tri(p):
        _validate_exact(p, {"lower", "mode", "upper"})
        low = _number(p, "lower")
        mode = _number(p, "mode")
        high = _number(p, "upper")
        if not low <= mode <= high or low >= high:
            raise ValueError("必须满足 lower ≤ mode ≤ upper 且 lower < upper")
        return {"lower": low, "mode": mode, "upper": high}
    def s_tri(n, p, r):
        low, mode, high = p["lower"], p["mode"], p["upper"]
        split = (mode - low) / (high - low)
        result = []
        for _ in range(n):
            u = _u(r)
            if u < split:
                result.append(low + math.sqrt(u * (high - low) * (mode - low)))
            else:
                result.append(high - math.sqrt((1.0 - u) * (high - low) * (high - mode)))
        return result
    plugins.append(_plugin(
        "triangular", "三角分布", "continuous",
        (_param("lower", "下界 a", "float", "最小取值", 0.0),
         _param("mode", "众数 c", "float", "峰值位置", 0.5),
         _param("upper", "上界 b", "float", "最大取值，需大于下界", 1.0)),
        v_tri, s_tri, "由下界、众数和上界确定的连续分布。"
    ))

    # 25. UniformInteger(lower, upper inclusive)
    def v_uint(p):
        _validate_exact(p, {"lower", "upper"})
        low = _integer(p, "lower")
        high = _integer(p, "upper")
        if low > high:
            raise ValueError("必须满足 lower ≤ upper")
        if high - low > 10**12:
            raise ValueError("整数均匀分布区间跨度不能超过 10^12")
        return {"lower": low, "upper": high}
    plugins.append(_plugin(
        "uniform_integer", "离散均匀分布", "discrete",
        (_param("lower", "整数下界 a", "int", "包含下界", 1),
         _param("upper", "整数上界 b", "int", "包含上界", 10)),
        v_uint,
        lambda n, p, r: [
            p["lower"] + int(r.random() * (p["upper"] - p["lower"] + 1))
            for _ in range(n)
        ],
        "在整数区间 [lower, upper] 上等概率取值，两个端点均包含。"
    ))

    # 26. UniformReal(lower, upper)
    def v_ureal(p):
        _validate_exact(p, {"lower", "upper"})
        low = _number(p, "lower")
        high = _number(p, "upper")
        if low >= high:
            raise ValueError("必须满足 lower < upper")
        return {"lower": low, "upper": high}
    plugins.append(_plugin(
        "uniform_real", "连续均匀分布", "continuous",
        (_param("lower", "下界 a", "float", "必须小于上界", 0.0),
         _param("upper", "上界 b", "float", "必须大于下界", 1.0)),
        v_ureal,
        lambda n, p, r: [
            p["lower"] + (p["upper"] - p["lower"]) * r.random()
            for _ in range(n)
        ],
        "在实数区间 [lower, upper] 上均匀取值。"
    ))

    # 27. Weibull(scale, shape)
    def v_weibull(p):
        _validate_exact(p, {"scale", "shape"})
        return {"scale": _positive(p, "scale"), "shape": _positive(p, "shape")}
    plugins.append(_plugin(
        "weibull", "Weibull 分布", "continuous",
        (_param("scale", "尺度参数 λ", "float", "必须大于 0", 1.0, 1e-15),
         _param("shape", "形状参数 k", "float", "必须大于 0", 2.0, 1e-15)),
        v_weibull,
        lambda n, p, r: [
            p["scale"] * (-math.log1p(-_u(r))) ** (1.0 / p["shape"])
            for _ in range(n)
        ],
        "采用尺度参数 λ 和形状参数 k 的 Weibull 分布。"
    ))

    # 28. Zipf(N, exponent)
    def v_zipf(p):
        _validate_exact(p, {"number_of_elements", "exponent"})
        return {
            "number_of_elements": _integer(p, "number_of_elements", minimum=1, maximum=200_000),
            "exponent": _positive(p, "exponent"),
        }
    def s_zipf(n, p, r):
        N, exponent = p["number_of_elements"], p["exponent"]
        weights = [1.0 / (k ** exponent) for k in range(1, N + 1)]
        total = sum(weights)
        cumulative = []
        running = 0.0
        for weight in weights:
            running += weight / total
            cumulative.append(running)
        cumulative[-1] = 1.0
        return [bisect.bisect_left(cumulative, r.random()) + 1 for _ in range(n)]
    plugins.append(_plugin(
        "zipf", "Zipf 分布", "discrete",
        (_param("number_of_elements", "元素总数 N", "int", "正整数，支持集为 1 到 N", 20, 1, 200_000),
         _param("exponent", "指数参数 s", "float", "必须大于 0", 1.5, 1e-15)),
        v_zipf, s_zipf,
        "有限支持集 k=1,...,N，概率正比于 1/k^s。"
    ))

    return plugins


def register_builtin_distributions(registry: DistributionRegistry) -> None:
    """注册老师清单中除 constant real 外的 27 种分布。"""
    registry.register(EnumeratedDistributionPlugin())
    for plugin in _make_plugins():
        registry.register(plugin)


def builtin_distribution_count() -> int:
    return 1 + len(_make_plugins())
