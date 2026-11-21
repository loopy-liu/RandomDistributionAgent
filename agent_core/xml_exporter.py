from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .contracts import SimulationResult


def _append_sample_value(parent: ET.Element, value: Any) -> None:
    """递归写入标量或向量样本，保留基本数据类型。"""
    if isinstance(value, (list, tuple)):
        vector = ET.SubElement(parent, "vector")
        for index, item in enumerate(value):
            component = ET.SubElement(vector, "component", index=str(index))
            _append_sample_value(component, item)
        return

    value_element = ET.SubElement(parent, "value")
    if isinstance(value, bool):
        value_element.set("type", "bool")
        value_element.text = "true" if value else "false"
    elif isinstance(value, int):
        value_element.set("type", "int")
        value_element.text = str(value)
    elif isinstance(value, float):
        value_element.set("type", "float")
        value_element.text = repr(value)
    else:
        value_element.set("type", "string")
        value_element.text = str(value)


def to_xml_element(result: SimulationResult) -> ET.Element:
    root = ET.Element(
        "simulation",
        schema_version="1.0",
        generated_at=result.generated_at,
    )

    distribution = ET.SubElement(
        root,
        "distribution",
        id=result.distribution_id,
        name=result.distribution_name,
    )

    parameters = ET.SubElement(distribution, "parameters")
    for name, value in result.parameters.items():
        parameter = ET.SubElement(parameters, "parameter", name=name)
        parameter.text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    sampling = ET.SubElement(
        root,
        "sampling",
        count=str(result.sample_count),
    )
    if result.seed is not None:
        sampling.set("seed", str(result.seed))

    samples_element = ET.SubElement(root, "samples", count=str(result.sample_count))
    for index, sample in enumerate(result.samples):
        sample_element = ET.SubElement(samples_element, "sample", index=str(index))
        _append_sample_value(sample_element, sample)

    return root


def export_xml(result: SimulationResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = to_xml_element(result)
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
