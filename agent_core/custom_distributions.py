from __future__ import annotations
import ast
import bisect
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .db_base import DatabaseAdapter
from .contracts import (
    DistributionMetadata,
    ParameterDefinition,
    RandomSource,
    SampleValue,
)
from .registry import DistributionRegistry


def _safe_factorial(value: float) -> int:
    if not float(value).is_integer() or value < 0 or value > 170:
        raise ValueError("factorial 的参数必须是 0 到 170 之间的整数")
    return math.factorial(int(value))


def _safe_comb(n: float, k: float) -> int:
    if not float(n).is_integer() or not float(k).is_integer():
        raise ValueError("comb 的两个参数必须是整数")
    n_int, k_int = int(n), int(k)
    if n_int < 0 or n_int > 10000 or k_int < 0 or k_int > n_int:
        raise ValueError("comb 要求 0 <= k <= n <= 10000")
    return math.comb(n_int, k_int)


_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "comb": _safe_comb,
    "factorial": _safe_factorial,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "ceil": math.ceil,
    "cos": math.cos,
    "cosh": math.cosh,
    "exp": math.exp,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "sinh": math.sinh,
    "sqrt": math.sqrt,
    "tan": math.tan,
    "tanh": math.tanh,
}

_ALLOWED_NAMES = {"x", "pi", "e", *_ALLOWED_FUNCTIONS.keys()}

_ALLOWED_BINARY_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}

_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class UnsafeExpressionError(ValueError):
    """表达式不在数学白名单范围内。"""


class SafeMathExpression:
    """受限数学表达式解释器，AST白名单解析，不执行任意代码"""

    def __init__(self, expression: str) -> None:
        if not isinstance(expression, str) or not expression.strip():
            raise UnsafeExpressionError("表达式不能为空")
        if len(expression) > 256:
            raise UnsafeExpressionError("表达式过长，最多允许 256 个字符")
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise UnsafeExpressionError("数学表达式语法不正确") from exc
        nodes = list(ast.walk(parsed))
        if len(nodes) > 80:
            raise UnsafeExpressionError("表达式过于复杂")
        self._tree = parsed.body
        self.expression = expression.strip()
        self._validate_node(self._tree)

    def _validate_node(self, node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise UnsafeExpressionError("只允许数字常量")
            return
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES:
                raise UnsafeExpressionError(f"不允许使用名称：{node.id}")
            return
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _ALLOWED_BINARY_OPERATORS:
                raise UnsafeExpressionError("只允许 +、-、*、/ 和 ** 运算")
            self._validate_node(node.left)
            self._validate_node(node.right)
            return
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _ALLOWED_UNARY_OPERATORS:
                raise UnsafeExpressionError("只允许一元正号和负号")
            self._validate_node(node.operand)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsafeExpressionError("不允许调用对象方法或属性")
            if node.func.id not in _ALLOWED_FUNCTIONS:
                raise UnsafeExpressionError(f"不允许使用函数：{node.func.id}")
            if node.keywords:
                raise UnsafeExpressionError("不允许使用关键字参数")
            if len(node.args) not in (1, 2):
                raise UnsafeExpressionError("数学函数参数数量不正确")
            for arg in node.args:
                self._validate_node(arg)
            return
        raise UnsafeExpressionError(f"表达式包含不支持的语法：{type(node).__name__}")

    def _evaluate(self, node: ast.AST, x: float) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id == "x":
                return float(x)
            if node.id == "pi":
                return math.pi
            if node.id == "e":
                return math.e
            raise UnsafeExpressionError(f"名称不能作为数值使用：{node.id}")
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand, x)
            return _ALLOWED_UNARY_OPERATORS[type(node.op)](value)
        if isinstance(node, ast.BinOp):
            left = self._evaluate(node.left, x)
            right = self._evaluate(node.right, x)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise UnsafeExpressionError("乘方指数绝对值不能超过 100")
            return _ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.Call):
            function = _ALLOWED_FUNCTIONS[node.func.id]
            args = [self._evaluate(arg, x) for arg in node.args]
            return float(function(*args))
        raise UnsafeExpressionError("表达式包含不支持的节点")

    def evaluate(self, x: float) -> float:
        try:
            value = float(self._evaluate(self._tree, float(x)))
        except (ArithmeticError, ValueError, OverflowError) as exc:
            raise ValueError(f"表达式在 x={x:g} 处无法计算：{exc}") from exc
        if not math.isfinite(value):
            raise ValueError(f"表达式在 x={x:g} 处产生非有限数值")
        return value


@dataclass(frozen=True)
class CustomDistributionDefinition:
    distribution_id: str
    name: str
    kind: str
    expression: str
    lower: float
    upper: float
    grid_size: int = 4097
    description: str = ""
    created_at: str = ""
    updated_at: str = ""


class CustomDistributionRepository:
    """自定义分布数据仓库，兼容SQLite/PostgreSQL"""

    def __init__(self, db) -> None:
        if isinstance(db, DatabaseAdapter):
            self.db = db
        else:
            self.db = DatabaseAdapter(database_path=db)
        self._initialize()

    def _adapt_sql(self, sql: str) -> str:
        """适配占位符：PostgreSQL用%s，SQLite用?"""
        if self.db.is_postgres:
            return sql
        return sql.replace("%s", "?")

    def _initialize(self) -> None:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            if self.db.is_postgres:
                sql = """
                    CREATE TABLE IF NOT EXISTS custom_distributions (
                        distribution_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        kind TEXT NOT NULL,
                        expression TEXT NOT NULL,
                        lower_bound DOUBLE PRECISION NOT NULL,
                        upper_bound DOUBLE PRECISION NOT NULL,
                        grid_size INTEGER NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """
            else:
                sql = """
                    CREATE TABLE IF NOT EXISTS custom_distributions (
                        distribution_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        kind TEXT NOT NULL,
                        expression TEXT NOT NULL,
                        lower_bound REAL NOT NULL,
                        upper_bound REAL NOT NULL,
                        grid_size INTEGER NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """
            cursor.execute(sql)

    def _check_name_exists(self, name: str, exclude_id: str | None = None) -> bool:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            if exclude_id:
                sql = "SELECT 1 FROM custom_distributions WHERE name = %s AND distribution_id != %s LIMIT 1"
                params = (name, exclude_id)
            else:
                sql = "SELECT 1 FROM custom_distributions WHERE name = %s LIMIT 1"
                params = (name,)
            cursor.execute(self._adapt_sql(sql), params)
            row = cursor.fetchone()
        return row is not None

    @staticmethod
    def _row_to_def(row) -> CustomDistributionDefinition:
        return CustomDistributionDefinition(
            distribution_id=row["distribution_id"],
            name=row["name"],
            kind=row["kind"],
            expression=row["expression"],
            lower=float(row["lower_bound"]),
            upper=float(row["upper_bound"]),
            grid_size=int(row["grid_size"]),
            description=row["description"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create(self, name, expression, lower, upper, grid_size=4097, description=""):
        normalized = _validate_definition(name, expression, lower, upper, grid_size, description)
        if self._check_name_exists(normalized["name"]):
            raise ValueError(f"分布名称「{normalized['name']}」已存在，请更换名称")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        distribution_id = f"custom_{uuid.uuid4().hex}"
        try:
            with self.db.connect() as conn:
                cursor = conn.cursor()
                sql = """
                    INSERT INTO custom_distributions
                    (distribution_id, name, kind, expression, lower_bound,
                     upper_bound, grid_size, description, created_at, updated_at)
                    VALUES (%s, %s, 'continuous_pdf', %s, %s, %s, %s, %s, %s, %s)
                """
                params = (
                    distribution_id, normalized["name"], normalized["expression"],
                    normalized["lower"], normalized["upper"], normalized["grid_size"],
                    normalized["description"], now, now,
                )
                cursor.execute(self._adapt_sql(sql), params)
        except Exception as exc:
            if "unique" in str(exc).lower():
                raise ValueError(f"分布名称「{normalized['name']}」已存在") from exc
            raise
        return self.get(distribution_id)

    def create_discrete(self, name, expression, lower, upper, description=""):
        normalized = _validate_discrete_definition(name, expression, lower, upper, description)
        if self._check_name_exists(normalized["name"]):
            raise ValueError(f"分布名称「{normalized['name']}」已存在，请更换名称")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        distribution_id = f"custom_{uuid.uuid4().hex}"
        try:
            with self.db.connect() as conn:
                cursor = conn.cursor()
                # 已修复：10个字段对应10个值，补上upper_bound
                sql = """
                    INSERT INTO custom_distributions
                    (distribution_id, name, kind, expression, lower_bound,
                     upper_bound, grid_size, description, created_at, updated_at)
                    VALUES (%s, %s, 'discrete_pmf', %s, %s, %s, 0, %s, %s, %s)
                """
                params = (
                    distribution_id, normalized["name"], normalized["expression"],
                    normalized["lower"], normalized["upper"],
                    normalized["description"], now, now,
                )
                cursor.execute(self._adapt_sql(sql), params)
        except Exception as exc:
            if "unique" in str(exc).lower():
                raise ValueError(f"分布名称「{normalized['name']}」已存在") from exc
            raise
        return self.get(distribution_id)

    def get(self, distribution_id: str):
        with self.db.connect() as conn:
            cursor = conn.cursor()
            sql = "SELECT * FROM custom_distributions WHERE distribution_id = %s"
            cursor.execute(self._adapt_sql(sql), (distribution_id,))
            row = cursor.fetchone()
        if row is None:
            raise KeyError(f"找不到自定义分布：{distribution_id}")
        return self._row_to_def(row)

    def list_all(self):
        with self.db.connect() as conn:
            cursor = conn.cursor()
            sql = "SELECT * FROM custom_distributions ORDER BY created_at, name"
            cursor.execute(self._adapt_sql(sql))
            rows = cursor.fetchall()
        return [self._row_to_def(row) for row in rows]

    def update(self, distribution_id, name, expression, lower, upper, grid_size=4097, description=""):
        normalized = _validate_definition(name, expression, lower, upper, grid_size, description)
        if self._check_name_exists(normalized["name"], exclude_id=distribution_id):
            raise ValueError(f"分布名称「{normalized['name']}」已被其他分布使用，请更换名称")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self.db.connect() as conn:
                cursor = conn.cursor()
                sql = """
                    UPDATE custom_distributions
                    SET name = %s, expression = %s, lower_bound = %s, upper_bound = %s,
                        grid_size = %s, description = %s, updated_at = %s
                    WHERE distribution_id = %s AND kind = 'continuous_pdf'
                """
                params = (
                    normalized["name"], normalized["expression"],
                    normalized["lower"], normalized["upper"], normalized["grid_size"],
                    normalized["description"], now, distribution_id,
                )
                cursor.execute(self._adapt_sql(sql), params)
                if cursor.rowcount == 0:
                    raise KeyError(f"找不到连续型自定义分布：{distribution_id}")
        except Exception as exc:
            if "unique" in str(exc).lower():
                raise ValueError(f"分布名称「{normalized['name']}」已被其他分布使用") from exc
            raise
        return self.get(distribution_id)

    def update_discrete(self, distribution_id, name, expression, lower, upper, description=""):
        normalized = _validate_discrete_definition(name, expression, lower, upper, description)
        if self._check_name_exists(normalized["name"], exclude_id=distribution_id):
            raise ValueError(f"分布名称「{normalized['name']}」已被其他分布使用，请更换名称")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self.db.connect() as conn:
                cursor = conn.cursor()
                sql = """
                    UPDATE custom_distributions
                    SET name = %s, kind = 'discrete_pmf', expression = %s,
                        lower_bound = %s, upper_bound = %s, grid_size = 0,
                        description = %s, updated_at = %s
                    WHERE distribution_id = %s AND kind = 'discrete_pmf'
                """
                params = (
                    normalized["name"], normalized["expression"],
                    normalized["lower"], normalized["upper"],
                    normalized["description"], now, distribution_id,
                )
                cursor.execute(self._adapt_sql(sql), params)
                if cursor.rowcount == 0:
                    raise KeyError(f"找不到离散型自定义分布：{distribution_id}")
        except Exception as exc:
            if "unique" in str(exc).lower():
                raise ValueError(f"分布名称「{normalized['name']}」已被其他分布使用") from exc
            raise
        return self.get(distribution_id)

    def delete(self, distribution_id: str) -> None:
        with self.db.connect() as conn:
            cursor = conn.cursor()
            sql = "DELETE FROM custom_distributions WHERE distribution_id = %s"
            cursor.execute(self._adapt_sql(sql), (distribution_id,))
            if cursor.rowcount == 0:
                raise KeyError(f"找不到自定义分布：{distribution_id}")


def _validate_definition(name, expression, lower, upper, grid_size, description):
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        raise ValueError("分布名称不能为空，且不能超过 80 个字符")
    if isinstance(lower, bool) or isinstance(upper, bool):
        raise ValueError("区间上下界必须是数字")
    try:
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError) as exc:
        raise ValueError("区间上下界必须是数字") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("必须满足有限数值 lower < upper")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int):
        raise ValueError("数值积分网格点数必须是整数")
    if not 1001 <= grid_size <= 20001:
        raise ValueError("数值积分网格点数必须在 1001 到 20001 之间")
    if not isinstance(description, str) or len(description) > 500:
        raise ValueError("描述不能超过 500 个字符")
    parsed = SafeMathExpression(expression)
    _build_density_table(parsed, lower, upper, grid_size)
    return {
        "name": name.strip(),
        "expression": parsed.expression,
        "lower": lower,
        "upper": upper,
        "grid_size": grid_size,
        "description": description.strip(),
    }


def calculate_pdf_total_area(expression: str, lower: float, upper: float, grid_size: int = 4097) -> float:
    parsed = SafeMathExpression(expression)
    _, _, _, total_area = _build_density_table(parsed, lower, upper, grid_size)
    return total_area


def _build_density_table(expression, lower, upper, grid_size):
    step = (upper - lower) / (grid_size - 1)
    xs = [lower + i * step for i in range(grid_size)]
    densities = []
    for x in xs:
        density = expression.evaluate(x)
        if density < -1e-12:
            raise ValueError(f"概率密度不能为负数：在 x={x:g} 处得到 {density:g}")
        densities.append(max(0.0, density))
    cumulative = [0.0]
    for index in range(1, grid_size):
        area = (densities[index - 1] + densities[index]) * 0.5 * step
        cumulative.append(cumulative[-1] + area)
    total_area = cumulative[-1]
    if not math.isfinite(total_area) or total_area <= 0:
        raise ValueError("概率密度在指定区间上的积分必须大于 0")
    return xs, densities, cumulative, total_area


def _validate_discrete_definition(name, expression, lower, upper, description):
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        raise ValueError("分布名称不能为空，且不能超过 80 个字符")
    if isinstance(lower, bool) or not isinstance(lower, int):
        raise ValueError("离散支持集下界必须是整数")
    if isinstance(upper, bool) or not isinstance(upper, int):
        raise ValueError("离散支持集上界必须是整数")
    if lower > upper:
        raise ValueError("离散支持集必须满足 lower <= upper")
    if upper - lower + 1 > 100_001:
        raise ValueError("离散支持集最多允许 100001 个整数")
    if not isinstance(description, str) or len(description) > 500:
        raise ValueError("描述不能超过 500 个字符")
    parsed = SafeMathExpression(expression)
    _build_probability_table(parsed, lower, upper)
    return {
        "name": name.strip(),
        "expression": parsed.expression,
        "lower": lower,
        "upper": upper,
        "description": description.strip(),
    }


def _build_probability_table(expression, lower, upper):
    values = list(range(lower, upper + 1))
    weights = []
    for value in values:
        weight = expression.evaluate(float(value))
        if weight < -1e-12:
            raise ValueError(f"概率质量函数不能为负数：在 x={value} 处得到 {weight:g}")
        weights.append(max(0.0, weight))
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("概率质量函数在指定整数支持集上的权重总和必须大于 0")
    probabilities = [weight / total for weight in weights]
    cumulative = []
    running = 0.0
    for probability in probabilities:
        running += probability
        cumulative.append(running)
    cumulative[-1] = 1.0
    return values, probabilities, cumulative


class CustomContinuousDistributionPlugin:
    """连续型自定义分布插件"""
    def __init__(self, definition: CustomDistributionDefinition) -> None:
        self.definition = definition
        self._expression = SafeMathExpression(definition.expression)
        self._table = _build_density_table(
            self._expression, definition.lower, definition.upper, definition.grid_size
        )

    @property
    def metadata(self) -> DistributionMetadata:
        return DistributionMetadata(
            distribution_id=self.definition.distribution_id,
            name=self.definition.name,
            kind="continuous",
            dimension="1D",
            version="custom-1.0",
            parameters=(
                ParameterDefinition(name="lower", label="区间下界", value_type="float", description="自定义概率密度函数的有限区间下界"),
                ParameterDefinition(name="upper", label="区间上界", value_type="float", description="自定义概率密度函数的有限区间上界"),
            ),
            description=f"自定义 PDF：{self.definition.expression}；定义区间 [{self.definition.lower}, {self.definition.upper}]",
        )

    def validate_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        if params:
            raise ValueError("该自定义分布的函数和区间已保存，无需额外仿真参数")
        return {
            "kind": "continuous_pdf",
            "expression": self.definition.expression,
            "lower": self.definition.lower,
            "upper": self.definition.upper,
            "grid_size": self.definition.grid_size,
        }

    def sample(self, n: int, params: dict[str, Any], rng: RandomSource) -> list[SampleValue]:
        xs, densities, cumulative, total_area = self._table
        samples = []
        for _ in range(n):
            target = rng.random() * total_area
            index = bisect.bisect_left(cumulative, target)
            if index <= 0:
                samples.append(xs[0])
                continue
            if index >= len(xs):
                samples.append(xs[-1])
                continue
            left_index = index - 1
            area_in_segment = target - cumulative[left_index]
            x0, x1 = xs[left_index], xs[index]
            y0, y1 = densities[left_index], densities[index]
            dx = x1 - x0
            slope = (y1 - y0) / dx
            if abs(slope) < 1e-14:
                distance = area_in_segment / y0 if y0 > 0 else 0.0
            else:
                discriminant = y0 * y0 + 2.0 * slope * area_in_segment
                discriminant = max(0.0, discriminant)
                distance = (-y0 + math.sqrt(discriminant)) / slope
            distance = min(dx, max(0.0, distance))
            samples.append(x0 + distance)
        return samples


class CustomDiscreteDistributionPlugin:
    """离散型自定义分布插件"""
    def __init__(self, definition: CustomDistributionDefinition) -> None:
        if definition.kind != "discrete_pmf":
            raise ValueError("CustomDiscreteDistributionPlugin 只接受 discrete_pmf")
        self.definition = definition
        self._expression = SafeMathExpression(definition.expression)
        self._table = _build_probability_table(
            self._expression, int(definition.lower), int(definition.upper)
        )

    @property
    def metadata(self) -> DistributionMetadata:
        return DistributionMetadata(
            distribution_id=self.definition.distribution_id,
            name=self.definition.name,
            kind="discrete",
            dimension="1D",
            version="custom-1.0",
            parameters=(),
            description=f"自定义 PMF：{self.definition.expression}；整数支持集 {{{int(self.definition.lower)}, ..., {int(self.definition.upper)}}}；权重在支持集上自动归一化。",
        )

    def validate_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        if params:
            raise ValueError("该自定义分布的 PMF 和整数支持集已保存，无需额外仿真参数")
        return {
            "kind": "discrete_pmf",
            "expression": self.definition.expression,
            "lower": int(self.definition.lower),
            "upper": int(self.definition.upper),
            "normalization": "sum_to_one",
        }

    def sample(self, n: int, params: dict[str, Any], rng: RandomSource) -> list[SampleValue]:
        values, _probabilities, cumulative = self._table
        samples = []
        for _ in range(n):
            index = bisect.bisect_left(cumulative, rng.random())
            if index >= len(values):
                index = len(values) - 1
            samples.append(values[index])
        return samples


class CustomDistributionManager:
    """自定义分布管理器，支持审计日志"""
    def __init__(self, repository, registry, audit_logger=None, operator: str = "system") -> None:
        self.repository = repository
        self.registry = registry
        self.audit_logger = audit_logger
        self.operator = operator
        self.reload_plugins()

    def _build_plugin(self, definition):
        if definition.kind == "continuous_pdf":
            return CustomContinuousDistributionPlugin(definition)
        if definition.kind == "discrete_pmf":
            return CustomDiscreteDistributionPlugin(definition)
        raise ValueError(f"不支持的自定义分布类型：{definition.kind}")

    def reload_plugins(self) -> None:
        for definition in self.repository.list_all():
            self.registry.unregister(definition.distribution_id)
            self.registry.register(self._build_plugin(definition))

    def set_operator(self, username: str) -> None:
        self.operator = username

    def create(self, **kwargs):
        definition = self.repository.create(**kwargs)
        self.registry.register(self._build_plugin(definition))
        if self.audit_logger:
            kind = "连续PDF" if definition.kind == "continuous_pdf" else "离散PMF"
            self.audit_logger.log(self.operator, "新增分布", definition.name, f"类型：{kind}，表达式：{definition.expression}")
        return definition

    def create_discrete(self, **kwargs):
        definition = self.repository.create_discrete(**kwargs)
        self.registry.register(self._build_plugin(definition))
        if self.audit_logger:
            self.audit_logger.log(self.operator, "新增分布", definition.name, f"类型：离散PMF，表达式：{definition.expression}")
        return definition

    def update(self, distribution_id: str, **kwargs):
        old_def = self.repository.get(distribution_id)
        definition = self.repository.update(distribution_id, **kwargs)
        self.registry.unregister(distribution_id)
        self.registry.register(self._build_plugin(definition))
        if self.audit_logger:
            self.audit_logger.log(self.operator, "修改分布", definition.name, f"原名称：{old_def.name}，新表达式：{definition.expression}")
        return definition

    def update_discrete(self, distribution_id: str, **kwargs):
        old_def = self.repository.get(distribution_id)
        definition = self.repository.update_discrete(distribution_id, **kwargs)
        self.registry.unregister(distribution_id)
        self.registry.register(self._build_plugin(definition))
        if self.audit_logger:
            self.audit_logger.log(self.operator, "修改分布", definition.name, f"原名称：{old_def.name}，新表达式：{definition.expression}")
        return definition

    def delete(self, distribution_id: str) -> None:
        definition = self.repository.get(distribution_id)
        self.repository.delete(distribution_id)
        self.registry.unregister(distribution_id)
        if self.audit_logger:
            kind = "连续PDF" if definition.kind == "continuous_pdf" else "离散PMF"
            self.audit_logger.log(self.operator, "删除分布", definition.name, f"类型：{kind}")
