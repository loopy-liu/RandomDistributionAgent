"""Trusted local extension example. Copy this file into the project extensions/ folder."""

from __future__ import annotations

from typing import Any

from agent_core.contracts import DistributionMetadata, ParameterDefinition, RandomSource, SampleValue


class DemoBernoulliPlugin:
    @property
    def metadata(self) -> DistributionMetadata:
        return DistributionMetadata(
            distribution_id="demo_bernoulli_extension",
            name="示例伯努利分布（外部插件）",
            kind="discrete",
            dimension="1D",
            version="1.0.0",
            parameters=(
                ParameterDefinition(
                    name="p",
                    label="成功概率 p",
                    value_type="float",
                    description="取值范围为 [0, 1]",
                    default=0.5,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
            description="用于演示外部 Python 插件自动发现机制。",
        )

    def validate_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        if set(params) != {"p"}:
            raise ValueError("参数必须且只能包含 p")
        p = params["p"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
            raise ValueError("p 必须是 [0, 1] 范围内的数字")
        return {"p": float(p)}

    def sample(self, n: int, params: dict[str, Any], rng: RandomSource) -> list[SampleValue]:
        p = params["p"]
        return [1 if rng.random() < p else 0 for _ in range(n)]


def build_plugin() -> DemoBernoulliPlugin:
    """The loader requires one build_plugin() factory per plugin file."""
    return DemoBernoulliPlugin()
