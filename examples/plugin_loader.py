from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .registry import DistributionRegistry


@dataclass(frozen=True)
class PluginLoadReport:
    loaded_ids: tuple[str, ...]
    errors: tuple[str, ...]


def _load_module(path: Path) -> ModuleType:
    module_name = f"_agent_extension_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("无法创建 Python 模块加载器")
    module = importlib.util.module_from_spec(spec)
    # dataclasses and some plugin libraries expect the module to exist in sys.modules.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _validate_plugin_shape(plugin: Any) -> None:
    for attribute in ("metadata", "validate_parameters", "sample"):
        if not hasattr(plugin, attribute):
            raise TypeError(f"插件缺少必要接口：{attribute}")
    metadata = plugin.metadata
    for attribute in ("distribution_id", "name", "kind", "dimension", "version", "parameters"):
        if not hasattr(metadata, attribute):
            raise TypeError(f"插件 metadata 缺少字段：{attribute}")
    if not callable(plugin.validate_parameters) or not callable(plugin.sample):
        raise TypeError("插件 validate_parameters 和 sample 必须是可调用方法")
    distribution_id = metadata.distribution_id
    if not isinstance(distribution_id, str) or not distribution_id.strip():
        raise TypeError("插件 distribution_id 必须是非空字符串")
    if distribution_id.startswith("custom_"):
        raise ValueError("custom_ 是数据库自定义分布保留前缀，外部插件不能使用")


def load_external_plugins(
    registry: DistributionRegistry,
    extension_dir: str | Path,
) -> PluginLoadReport:
    """Load trusted developer-authored plugin_*.py files from one fixed folder.

    Python extension files execute as code when imported. Therefore this loader is
    intentionally limited to the local, administrator-managed extensions folder;
    it must never be pointed at user uploads or files supplied through the public UI.
    """
    directory = Path(extension_dir)
    directory.mkdir(parents=True, exist_ok=True)
    resolved_directory = directory.resolve()
    loaded: list[str] = []
    errors: list[str] = []

    for path in sorted(directory.glob("plugin_*.py")):
        if path.is_symlink() or not path.is_file():
            errors.append(f"{path.name}：跳过符号链接或非普通文件")
            continue
        resolved_path = path.resolve()
        if resolved_path.parent != resolved_directory:
            errors.append(f"{path.name}：插件路径不在指定目录内")
            continue
        try:
            module = _load_module(resolved_path)
            factory = getattr(module, "build_plugin", None)
            if not callable(factory):
                raise TypeError("插件文件必须提供 build_plugin() 工厂函数")
            plugin = factory()
            _validate_plugin_shape(plugin)
            registry.register(plugin)
            loaded.append(plugin.metadata.distribution_id)
        except Exception as exc:
            errors.append(f"{path.name}：{type(exc).__name__}: {exc}")

    return PluginLoadReport(tuple(loaded), tuple(errors))
