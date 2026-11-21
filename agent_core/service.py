from __future__ import annotations

import random
from typing import Any

from .contracts import SimulationResult
from .registry import DistributionRegistry


class SimulationService:
    """统一仿真入口；不包含任何具体分布的数学实现。"""

    def __init__(
        self,
        registry: DistributionRegistry,
        max_samples: int = 100_000,
    ) -> None:
        if max_samples < 1:
            raise ValueError("max_samples 必须是正整数")
        self._registry = registry
        self._max_samples = max_samples

    def simulate(
        self,
        distribution_id: str,
        params: dict[str, Any],
        sample_count: int,
        seed: int | None = None,
    ) -> SimulationResult:
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise ValueError("样本数量必须是整数")
        if sample_count <= 0:
            raise ValueError("样本数量必须大于 0")
        if sample_count > self._max_samples:
            raise ValueError(
                f"单次样本数量不能超过 {self._max_samples}"
            )
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise ValueError("随机种子必须是整数或留空")

        plugin = self._registry.get(distribution_id)
        normalized = plugin.validate_parameters(dict(params))
        rng = random.Random(seed)
        samples = plugin.sample(sample_count, normalized, rng)

        if len(samples) != sample_count:
            raise RuntimeError(
                f"插件 {distribution_id} 返回了 {len(samples)} 个样本，"
                f"预期为 {sample_count} 个"
            )

        return SimulationResult.create(
            distribution_id=plugin.metadata.distribution_id,
            distribution_name=plugin.metadata.name,
            parameters=normalized,
            sample_count=sample_count,
            seed=seed,
            samples=samples,
        )
