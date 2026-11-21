from __future__ import annotations

from typing import Protocol, Any

from .contracts import DistributionMetadata, RandomSource, SampleValue


class DistributionPlugin(Protocol):
    @property
    def metadata(self) -> DistributionMetadata:
        ...

    def validate_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """验证参数并返回规范化后的新字典；不得修改调用方原字典。"""
        ...

    def sample(
        self,
        n: int,
        params: dict[str, Any],
        rng: RandomSource,
    ) -> list[SampleValue]:
        ...


class DistributionRegistry:
    """分布注册中心。新增普通分布时注册插件，不修改仿真服务。"""

    def __init__(self) -> None:
        self._plugins: dict[str, DistributionPlugin] = {}

    def register(self, plugin: DistributionPlugin) -> None:
        distribution_id = plugin.metadata.distribution_id.strip()
        if not distribution_id:
            raise ValueError("分布 ID 不能为空")
        if distribution_id in self._plugins:
            raise ValueError(f"分布 ID 已注册：{distribution_id}")
        self._plugins[distribution_id] = plugin

    def get(self, distribution_id: str) -> DistributionPlugin:
        try:
            return self._plugins[distribution_id]
        except KeyError as exc:
            raise KeyError(f"未注册的分布：{distribution_id}") from exc

    def unregister(self, distribution_id: str) -> None:
        """移除已注册的分布插件；不存在时不报错。"""
        self._plugins.pop(distribution_id, None)

    def list_metadata(self) -> list[DistributionMetadata]:
        return [plugin.metadata for plugin in self._plugins.values()]

    def __len__(self) -> int:
        return len(self._plugins)
