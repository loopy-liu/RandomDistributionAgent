from __future__ import annotations

import math
from typing import Any

from ..contracts import (
    DistributionMetadata,
    ParameterDefinition,
    RandomSource,
    SampleValue,
)


class EnumeratedDistributionPlugin:
    """
    枚举分布插件。

    抽样核心沿用基础作业的“累积概率区间映射”方法：
    先计算累积概率，再将均匀随机数映射到对应枚举值。
    """

    @property
    def metadata(self) -> DistributionMetadata:
        return DistributionMetadata(
            distribution_id="enumerated",
            name="枚举分布",
            kind="discrete",
            dimension="1D",
            version="1.0.0",
            parameters=(
                ParameterDefinition(
                    name="values",
                    label="枚举值",
                    value_type="list[number|string]",
                    description="有限个互不重复的候选取值",
                    default=[1, 2, 3],
                ),
                ParameterDefinition(
                    name="probabilities",
                    label="对应概率",
                    value_type="list[float]",
                    description="每个枚举值对应的概率，非负且总和为 1",
                    default=[0.2, 0.3, 0.5],
                ),
            ),
            description="按照给定概率从有限个枚举值中随机抽样。",
        )

    def validate_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        values = params.get("values")
        probabilities = params.get("probabilities")

        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError("枚举值 values 必须是非空列表")
        if not isinstance(probabilities, (list, tuple)):
            raise ValueError("概率 probabilities 必须是列表")
        if len(values) != len(probabilities):
            raise ValueError("枚举值和概率的数量必须一致")

        normalized_values: list[SampleValue] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError("枚举值只支持整数、浮点数或字符串")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("枚举值不能是 NaN 或无穷大")
            normalized_values.append(value)

        try:
            if len(set(normalized_values)) != len(normalized_values):
                raise ValueError("枚举值不能重复")
        except TypeError as exc:
            raise ValueError("枚举值必须是可比较的基本类型") from exc

        normalized_probabilities: list[float] = []
        for probability in probabilities:
            if isinstance(probability, bool) or not isinstance(
                probability, (int, float)
            ):
                raise ValueError("概率必须是数字")
            p = float(probability)
            if not math.isfinite(p) or p < 0:
                raise ValueError("概率必须是有限的非负数字")
            normalized_probabilities.append(p)

        if not math.isclose(
            sum(normalized_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError("所有概率之和必须等于 1")

        return {
            "values": normalized_values,
            "probabilities": normalized_probabilities,
        }

    def sample(
        self,
        n: int,
        params: dict[str, Any],
        rng: RandomSource,
    ) -> list[SampleValue]:
        values = params["values"]
        probabilities = params["probabilities"]

        cumulative: list[float] = []
        total = 0.0
        for probability in probabilities:
            total += probability
            cumulative.append(total)
        cumulative[-1] = 1.0

        samples: list[SampleValue] = []
        for _ in range(n):
            u = rng.random()
            for index, upper_bound in enumerate(cumulative):
                if u < upper_bound:
                    samples.append(values[index])
                    break
            else:
                # 防御浮点累计误差；最后一个区间吸收尾部。
                samples.append(values[-1])

        return samples
