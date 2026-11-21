from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, TypeAlias


SampleValue: TypeAlias = int | float | str | bool | list[Any]


class RandomSource(Protocol):
    """插件使用的最小随机数接口；random.Random 符合该接口。"""

    def random(self) -> float:
        ...


@dataclass(frozen=True)
class ParameterDefinition:
    name: str
    label: str
    value_type: str
    description: str
    required: bool = True
    default: Any = None
    minimum: float | int | None = None
    maximum: float | int | None = None


@dataclass(frozen=True)
class DistributionMetadata:
    distribution_id: str
    name: str
    kind: str
    dimension: str
    version: str
    parameters: tuple[ParameterDefinition, ...]
    description: str = ""


@dataclass(frozen=True)
class SimulationResult:
    distribution_id: str
    distribution_name: str
    parameters: dict[str, Any]
    sample_count: int
    seed: int | None
    samples: list[SampleValue]
    generated_at: str

    @property
    def sample_shape(self) -> tuple[int, ...]:
        if not self.samples:
            return (0,)
        first = self.samples[0]
        if isinstance(first, list):
            return (len(self.samples), len(first))
        return (len(self.samples),)

    @classmethod
    def create(
        cls,
        distribution_id: str,
        distribution_name: str,
        parameters: dict[str, Any],
        sample_count: int,
        seed: int | None,
        samples: list[SampleValue],
    ) -> "SimulationResult":
        return cls(
            distribution_id=distribution_id,
            distribution_name=distribution_name,
            parameters=parameters,
            sample_count=sample_count,
            seed=seed,
            samples=samples,
            generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
