from __future__ import annotations
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager
import streamlit as st
from agent_core.db_base import DatabaseAdapter
from agent_core.custom_distributions import (
    CustomDistributionManager,
    CustomDistributionRepository,
    calculate_pdf_total_area,
)
from agent_core.auth import UserRepository
from agent_core.audit_log import AuditLogRepository
from agent_core.plugins.builtins import register_builtin_distributions
from agent_core.plugin_loader import load_external_plugins
from agent_core.registry import DistributionRegistry
from agent_core.service import SimulationService
from agent_core.xml_exporter import to_xml_element

BASE_DIR = Path(__file__).parent
DATABASE_PATH = BASE_DIR / "data" / "agent.sqlite3"
EXTENSIONS_DIR = BASE_DIR / "extensions"
MAX_SAMPLES = 100_000


# ============== 全局视觉美化 ==============
def apply_custom_style():
    st.markdown(
        """
        <style>
            /* 主背景：柔蓝渐变 */
            .stApp {
                background: linear-gradient(180deg, #f8fbff 0%, #f0f7ff 100%);
            }

            /* 侧边栏背景：同色系 */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f5f9ff 0%, #eef5ff 100%);
                border-right: 1px solid #e0e9f5;
            }

            /* 主按钮统一蓝色 */
            button[kind="primary"] {
                background-color: #1677ff !important;
                border-color: #1677ff !important;
                border-radius: 8px !important;
            }
            button[kind="primary"]:hover {
                background-color: #0958d9 !important;
                border-color: #0958d9 !important;
            }

            /* 标签页蓝色下划线 */
            [data-baseweb="tab-highlight"] {
                background-color: #1677ff !important;
            }
            [aria-selected="true"] {
                color: #1677ff !important;
            }

            /* 输入框圆角 */
            input, textarea, [data-baseweb="select"] {
                border-radius: 8px !important;
            }

            /* 表单卡片白色悬浮 */
            [data-testid="stForm"] {
                background: #ffffff;
                border-radius: 12px;
                border: 1px solid #eef2f7;
                box-shadow: 0 2px 12px rgba(22, 40, 70, 0.05);
                padding: 20px;
            }

            /* 底部状态条/加载条改为蓝色 */
            [data-testid="stStatusWidget"] {
                background-color: #1677ff !important;
            }

            /* 顶部留白 */
            .block-container {
                padding-top: 2rem !important;
            }

            /* 强制修改侧边栏单选圆点为蓝色（覆盖SVG） */
            [data-testid="stSidebar"] [data-testid="stRadio"] svg circle {
                stroke: #1677ff !important;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] svg path {
                fill: #1677ff !important;
                stroke: #1677ff !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_matplotlib_chinese_font() -> str | None:
    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    font_dir = windows_dir / "Fonts"
    font_files = ["msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf"]
    for filename in font_files:
        font_path = font_dir / filename
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
                font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
                plt.rcParams["font.family"] = font_name
                plt.rcParams["axes.unicode_minus"] = False
                return font_name
            except (OSError, RuntimeError, ValueError):
                continue
    for font_name in (
        "Microsoft YaHei", "SimHei", "DengXian",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei",
    ):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=font_name),
                fallback_to_default=False,
            )
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
        except (ValueError, RuntimeError):
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return None


CHINESE_FONT_NAME = configure_matplotlib_chinese_font()


def build_runtime():
    db = DatabaseAdapter(DATABASE_PATH)
    repository = CustomDistributionRepository(db)
    user_repo = UserRepository(db)
    audit_repo = AuditLogRepository(db)
    registry = DistributionRegistry()
    register_builtin_distributions(registry)
    extension_report = load_external_plugins(registry, EXTENSIONS_DIR)
    custom_manager = CustomDistributionManager(repository, registry, audit_logger=audit_repo)
    simulation_service = SimulationService(registry, max_samples=MAX_SAMPLES)
    return repository, user_repo, audit_repo, registry, custom_manager, simulation_service, extension_report


def init_auth_state():
    if "is_logged_in" not in st.session_state:
        st.session_state["is_logged_in"] = False
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = None
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = None


def render_sidebar_auth(user_repo: UserRepository):
    with st.sidebar:
        st.subheader("账号中心")
        if st.session_state["is_logged_in"]:
            role_label = "管理员" if st.session_state["user_role"] == "admin" else "普通成员"
            st.success(f"已登录：{st.session_state['current_user']}")
            st.caption(f"角色：{role_label}")
            if st.button("退出登录", use_container_width=True):
                st.session_state["is_logged_in"] = False
                st.session_state["current_user"] = None
                st.session_state["user_role"] = None
                st.rerun()
            return

        auth_mode = st.radio("操作", ["登录", "注册"], horizontal=True)
        username = st.text_input("用户名", key="auth_username")
        password = st.text_input("密码", type="password", key="auth_password")

        if auth_mode == "登录":
            if st.button("登录", type="primary", use_container_width=True):
                try:
                    user = user_repo.verify(username, password)
                    st.session_state["is_logged_in"] = True
                    st.session_state["current_user"] = user.username
                    st.session_state["user_role"] = user.role
                    st.success("登录成功")
                    st.rerun()
                except Exception as exc:
                    st.error(f"登录失败：{exc}")
        else:
            if st.button("注册账号", type="secondary", use_container_width=True):
                try:
                    user_repo.register(username, password)
                    st.success("注册成功，请登录")
                except Exception as exc:
                    st.error(f"注册失败：{exc}")


def parse_json_parameter(raw: str, label: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} 必须是合法 JSON，例如 [0, 0]") from exc


def parse_scalar_list(raw: str) -> list[int | float | str]:
    items = [item.strip() for item in raw.replace("，", ",").split(",") if item.strip()]
    if not items:
        raise ValueError("请至少输入一个枚举值")
    parsed = []
    for item in items:
        try:
            if item.lstrip("+-").isdigit():
                parsed.append(int(item))
            else:
                value = float(item)
                if not math.isfinite(value):
                    raise ValueError
                parsed.append(value)
        except ValueError:
            parsed.append(item)
    return parsed


def parse_float_list(raw: str) -> list[float]:
    items = [item.strip() for item in raw.replace("，", ",").split(",") if item.strip()]
    if not items:
        raise ValueError("请至少输入一个概率")
    try:
        values = [float(item) for item in items]
    except ValueError as exc:
        raise ValueError("概率列表中包含无法识别的数字") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError("概率必须是有限数字")
    return values


def render_parameter_input(parameter):
    key = parameter.name
    label = parameter.label
    help_text = parameter.description
    default = parameter.default
    if parameter.value_type == "int":
        kwargs = {"label": label, "value": int(default), "help": help_text, "key": f"param_{key}"}
        if parameter.minimum is not None:
            kwargs["min_value"] = int(parameter.minimum)
        if parameter.maximum is not None:
            kwargs["max_value"] = int(parameter.maximum)
        return st.number_input(**kwargs)
    if parameter.value_type == "float":
        kwargs = {
            "label": label, "value": float(default), "help": help_text,
            "key": f"param_{key}", "format": "%.8g",
        }
        if parameter.minimum is not None:
            kwargs["min_value"] = float(parameter.minimum)
        if parameter.maximum is not None:
            kwargs["max_value"] = float(parameter.maximum)
        return st.number_input(**kwargs)
    if parameter.value_type in ("vector_float", "matrix_float"):
        raw_default = json.dumps(default, ensure_ascii=False)
        raw = st.text_area(
            label, value=raw_default, help=f"{help_text}。请按 JSON 格式输入。",
            key=f"param_{key}", height=100 if parameter.value_type == "matrix_float" else 60,
        )
        return parse_json_parameter(raw, label)
    if parameter.value_type == "list[number|string]":
        raw_default = ", ".join(str(item) for item in default)
        raw = st.text_input(label, value=raw_default, help=help_text, key=f"param_{key}")
        return parse_scalar_list(raw)
    if parameter.value_type == "list[float]":
        raw_default = ", ".join(str(item) for item in default)
        raw = st.text_input(label, value=raw_default, help=help_text, key=f"param_{key}")
        return parse_float_list(raw)
    raise ValueError(f"界面尚不支持参数类型：{parameter.value_type}")


def _result_dataframe_data(samples):
    if samples and isinstance(samples[0], list):
        dimension = len(samples[0])
        data = {"样本序号": list(range(1, len(samples) + 1))}
        for index in range(dimension):
            data[f"X{index + 1}"] = [row[index] for row in samples]
        return data
    return {"样本序号": list(range(1, len(samples) + 1)), "样本值": samples}


def show_results(result) -> None:
    st.subheader("仿真结果")
    metric_cols = st.columns(3)
    metric_cols[0].metric("分布", result.distribution_name)
    metric_cols[1].metric("样本数量", f"{result.sample_count:,}")
    metric_cols[2].metric("随机种子", "未固定" if result.seed is None else str(result.seed))
    samples = result.samples
    preview_count = min(len(samples), 1000)
    st.caption(f"下方展示前 {preview_count:,} 个样本；完整数据可通过 XML 文件导出。")
    preview = samples[:preview_count]
    st.dataframe(_result_dataframe_data(preview), use_container_width=True, height=260)
    left, right = st.columns(2)
    is_vector = bool(samples) and isinstance(samples[0], list)
    numeric = not is_vector and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in samples
    )
    with left:
        st.markdown("**样本点图**")
        fig, ax = plt.subplots(figsize=(7, 4))
        if is_vector:
            dimension = len(samples[0])
            if dimension >= 2:
                ax.scatter(
                    [row[0] for row in samples], [row[1] for row in samples],
                    s=12, alpha=0.55, color="#1677ff",
                )
                ax.set_xlabel("X1")
                ax.set_ylabel("X2")
                ax.set_title("前两个维度的联合样本点")
            else:
                ax.scatter(range(1, len(samples) + 1), [row[0] for row in samples], s=12, alpha=0.55, color="#1677ff")
                ax.set_xlabel("样本序号")
                ax.set_ylabel("X1")
        elif numeric:
            ax.scatter(range(1, len(samples) + 1), samples, s=12, alpha=0.55, color="#1677ff")
            ax.set_xlabel("样本序号")
            ax.set_ylabel("样本值")
        else:
            categories = list(dict.fromkeys(samples))
            mapping = {value: index for index, value in enumerate(categories)}
            y = [mapping[value] for value in samples]
            ax.scatter(range(1, len(samples) + 1), y, s=12, alpha=0.55, color="#1677ff")
            ax.set_yticks(range(len(categories)), [str(value) for value in categories])
            ax.set_xlabel("样本序号")
            ax.set_ylabel("枚举值")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("**样本频数 / 边际分布图**")
        fig, ax = plt.subplots(figsize=(7, 4))
        if is_vector:
            first_dimension = [row[0] for row in samples]
            ax.hist(first_dimension, bins=min(40, max(8, int(math.sqrt(len(samples))))), edgecolor="white", color="#1677ff", alpha=0.85)
            ax.set_xlabel("X1")
            ax.set_ylabel("频数")
            ax.set_title("第一维边际频数")
        elif numeric:
            bin_count = min(40, max(8, int(math.sqrt(len(samples)))))
            ax.hist(samples, bins=bin_count, edgecolor="white", color="#1677ff", alpha=0.85)
            ax.set_xlabel("样本值")
            ax.set_ylabel("频数")
        else:
            categories = list(dict.fromkeys(samples))
            counts = [samples.count(value) for value in categories]
            ax.bar([str(value) for value in categories], counts, color="#1677ff", alpha=0.85)
            ax.set_xlabel("枚举值")
            ax.set_ylabel("频数")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    xml_bytes = ET.tostring(to_xml_element(result), encoding="utf-8", xml_declaration=True)
    st.download_button(
        label="下载本次仿真的 XML 文件",
        data=xml_bytes,
        file_name=f"{result.distribution_id}_simulation.xml",
        mime="application/xml",
        use_container_width=True,
    )


def render_simulation_workspace(registry, custom_definitions, simulation_service) -> None:
    st.header("随机分布仿真工作台")
    st.write("选择分布、填写参数和样本数量，生成样本数据、样本点图并导出 XML。")
    all_metadata = registry.list_metadata()
    builtin_metadata = [m for m in all_metadata if not m.distribution_id.startswith("custom_")]
    custom_metadata = [m for m in all_metadata if m.distribution_id.startswith("custom_")]
    st.markdown("**内置分布**")
    builtin_ids = [m.distribution_id for m in builtin_metadata]
    labels = {m.distribution_id: m.name for m in builtin_metadata}
    custom_ids = [m.distribution_id for m in custom_metadata]
    labels.update({m.distribution_id: f"{m.name}（自定义）" for m in custom_metadata})
    selected_id = st.selectbox(
        "选择仿真分布",
        options=builtin_ids + custom_ids,
        format_func=lambda value: labels[value],
    )
    selected_metadata = next(m for m in all_metadata if m.distribution_id == selected_id)
    with st.form("simulation_form"):
        params: dict[str, Any] = {}
        if selected_id.startswith("custom_"):
            st.caption("该自定义分布的表达式和定义区间已保存，无需重复填写。")
        elif selected_metadata.parameters:
            st.markdown("**分布参数**")
            columns = st.columns(2)
            for index, parameter in enumerate(selected_metadata.parameters):
                with columns[index % 2]:
                    params[parameter.name] = render_parameter_input(parameter)
        sample_count = st.number_input(
            "样本点个数", min_value=1, max_value=MAX_SAMPLES, value=1000, step=100,
        )
        seed_text = st.text_input("随机种子（可留空）", value="2026")
        submitted = st.form_submit_button("开始仿真", type="primary", use_container_width=True)
    if submitted:
        try:
            seed = None if not seed_text.strip() else int(seed_text.strip())
            result = simulation_service.simulate(
                distribution_id=selected_id, params=params,
                sample_count=int(sample_count), seed=seed,
            )
            st.session_state["latest_result"] = result
            st.success("仿真完成。")
        except Exception as exc:
            st.error(f"仿真失败：{exc}")
    result = st.session_state.get("latest_result")
    if result is not None:
        show_results(result)


def render_custom_distribution_manager(manager, repository) -> None:
    st.header("自定义分布管理")
    if not st.session_state["is_logged_in"]:
        st.info("请在左侧侧边栏登录账号后，再使用自定义分布管理功能")
        st.caption("未登录用户可正常使用仿真工作台的全部功能")
        return

    manager.set_operator(st.session_state["current_user"])
    st.write("支持通过数学表达式创建连续型 PDF 或离散型 PMF。定义保存在数据库中，并通过统一插件接口参与仿真。")
    with st.expander("表达式支持范围与校验规则", expanded=False):
        st.write("变量：`x`；常量：`pi`、`e`。")
        st.write("运算：`+`、`-`、`*`、`/`、`**`。")
        st.write("函数：`abs`、`sqrt`、`exp`、`log`、`log10`、`sin`、`cos`、`tan`、`factorial`、`comb` 等。")
        st.info("**连续型 PDF**：系统自动归一化采样，表达式作为相对密度形状；保存时会显示区间积分值，严格 PDF 要求积分接近 1。")
        st.info("**离散型 PMF**：表达式作为非负权重，系统会在整数支持集上自动归一化为概率。")
        st.warning(
            "表达式由 AST 白名单解释，不执行任意 Python 代码。Python 插件扩展仅供可信开发者放入本地 extensions 目录，"
            "不要允许公网用户上传并执行 Python 文件。"
        )

    st.subheader("新增自定义分布")
    creation_kind = st.radio(
        "分布类型", options=["连续型 PDF", "离散型 PMF"],
        horizontal=True, key="custom_creation_kind",
    )
    if creation_kind == "连续型 PDF":
        with st.form("create_custom_pdf"):
            new_name = st.text_input("分布名称", placeholder="例如：递增密度分布")
            new_expression = st.text_input("概率密度函数 f(x)", value="2*x")
            col1, col2 = st.columns(2)
            with col1:
                new_lower = st.number_input("区间下界", value=0.0, format="%.6f")
            with col2:
                new_upper = st.number_input("区间上界", value=1.0, format="%.6f")
            new_grid = st.number_input(
                "数值积分网格点数", min_value=1001, max_value=20001,
                value=4097, step=1000,
            )
            new_description = st.text_input("描述（可选）")
            create_submitted = st.form_submit_button("保存新分布", type="primary", use_container_width=True)
        if create_submitted:
            try:
                created = manager.create(
                    name=new_name, expression=new_expression,
                    lower=new_lower, upper=new_upper,
                    grid_size=int(new_grid), description=new_description,
                )
                area = calculate_pdf_total_area(new_expression, new_lower, new_upper, int(new_grid))
                st.success(f"已创建：{created.name}")
                st.caption(f"该表达式在区间上的积分值为：{area:.4f}；系统将自动归一化后采样")
                st.rerun()
            except Exception as exc:
                st.error(f"创建失败：{exc}")
    else:
        with st.form("create_custom_pmf"):
            new_name = st.text_input("分布名称", placeholder="例如：自定义泊松型分布")
            new_expression = st.text_input(
                "概率质量权重函数 p(x)",
                value="exp(-4)*4**x/factorial(x)",
                help="x 表示整数取值；系统会在下方支持集上计算权重并自动归一化。",
            )
            col1, col2 = st.columns(2)
            with col1:
                new_lower = st.number_input("整数支持集下界", value=0, step=1, format="%d")
            with col2:
                new_upper = st.number_input("整数支持集上界", value=15, step=1, format="%d")
            new_description = st.text_input("描述（可选）")
            create_submitted = st.form_submit_button("保存新分布", type="primary", use_container_width=True)
        if create_submitted:
            try:
                created = manager.create_discrete(
                    name=new_name, expression=new_expression,
                    lower=int(new_lower), upper=int(new_upper),
                    description=new_description,
                )
                st.success(f"已创建：{created.name}")
                st.rerun()
            except Exception as exc:
                st.error(f"创建失败：{exc}")

    st.divider()
    st.subheader("已保存的自定义分布")
    definitions = repository.list_all()
    if not definitions:
        st.info("目前还没有自定义分布。可以先创建一个 PDF 或 PMF 示例。")
        return

    selected_custom_id = st.selectbox(
        "选择要查看或修改的分布",
        options=[item.distribution_id for item in definitions],
        format_func=lambda value: next(
            f"{item.name}（{'PDF' if item.kind == 'continuous_pdf' else 'PMF'} · {item.distribution_id[-8:]}）"
            for item in definitions if item.distribution_id == value
        ),
    )
    selected = repository.get(selected_custom_id)

    if selected.kind == "continuous_pdf":
        st.code(f"f(x) = {selected.expression}\nx ∈ [{selected.lower}, {selected.upper}]", language="text")
        try:
            area = calculate_pdf_total_area(selected.expression, selected.lower, selected.upper, selected.grid_size)
            st.caption(f"区间积分值：{area:.4f}（系统自动归一化采样；严格 PDF 要求积分接近 1）")
        except Exception:
            pass
        if selected.description:
            st.caption(selected.description)
        with st.form(f"edit_custom_pdf_{selected_custom_id}"):
            edit_name = st.text_input("分布名称", value=selected.name)
            edit_expression = st.text_input("概率密度函数 f(x)", value=selected.expression)
            col1, col2 = st.columns(2)
            with col1:
                edit_lower = st.number_input(
                    "区间下界", value=float(selected.lower), format="%.6f",
                    key=f"edit_pdf_lower_{selected_custom_id}",
                )
            with col2:
                edit_upper = st.number_input(
                    "区间上界", value=float(selected.upper), format="%.6f",
                    key=f"edit_pdf_upper_{selected_custom_id}",
                )
            edit_grid = st.number_input(
                "数值积分网格点数", min_value=1001, max_value=20001,
                value=int(selected.grid_size), step=1000,
                key=f"edit_pdf_grid_{selected_custom_id}",
            )
            edit_description = st.text_input(
                "描述", value=selected.description,
                key=f"edit_pdf_description_{selected_custom_id}",
            )
            update_submitted = st.form_submit_button("保存修改", type="primary", use_container_width=True)
        if update_submitted:
            try:
                manager.update(
                    selected_custom_id, name=edit_name, expression=edit_expression,
                    lower=edit_lower, upper=edit_upper,
                    grid_size=int(edit_grid), description=edit_description,
                )
                st.success("修改已保存。")
                st.rerun()
            except Exception as exc:
                st.error(f"修改失败：{exc}")
    elif selected.kind == "discrete_pmf":
        st.code(
            f"p(x) ∝ {selected.expression}\nx ∈ {{{int(selected.lower)}, ..., {int(selected.upper)}}}",
            language="text",
        )
        st.caption("权重在指定整数支持集上自动归一化。")
        if selected.description:
            st.caption(selected.description)
        with st.form(f"edit_custom_pmf_{selected_custom_id}"):
            edit_name = st.text_input("分布名称", value=selected.name)
            edit_expression = st.text_input("概率质量权重函数 p(x)", value=selected.expression)
            col1, col2 = st.columns(2)
            with col1:
                edit_lower = st.number_input(
                    "整数支持集下界", value=int(selected.lower), step=1, format="%d",
                    key=f"edit_pmf_lower_{selected_custom_id}",
                )
            with col2:
                edit_upper = st.number_input(
                    "整数支持集上界", value=int(selected.upper), step=1, format="%d",
                    key=f"edit_pmf_upper_{selected_custom_id}",
                )
            edit_description = st.text_input(
                "描述", value=selected.description,
                key=f"edit_pmf_description_{selected_custom_id}",
            )
            update_submitted = st.form_submit_button("保存修改", type="primary", use_container_width=True)
        if update_submitted:
            try:
                manager.update_discrete(
                    selected_custom_id, name=edit_name, expression=edit_expression,
                    lower=int(edit_lower), upper=int(edit_upper),
                    description=edit_description,
                )
                st.success("修改已保存。")
                st.rerun()
            except Exception as exc:
                st.error(f"修改失败：{exc}")
    else:
        st.error(f"数据库中存在暂不支持的自定义类型：{selected.kind}")

    st.divider()
    col_del1, col_del2 = st.columns([1, 1])
    with col_del1:
        confirm_delete = st.checkbox(
            "确认删除该分布（删除后无法恢复）",
            key=f"confirm_del_{selected_custom_id}",
        )
    with col_del2:
        if st.button(
            "删除此自定义分布", type="secondary", disabled=not confirm_delete,
            key=f"delete_custom_{selected_custom_id}",
        ):
            try:
                manager.delete(selected_custom_id)
                st.success("已删除该自定义分布。")
                st.rerun()
            except Exception as exc:
                st.error(f"删除失败：{exc}")


def render_about_page(user_repo, audit_repo, extension_report) -> None:
    st.subheader("当前版本 V1.0 功能范围")
    st.markdown(
        """
        - 内置分布：课程清单中除 constant real 外的 27 种分布。
        - 自定义分布：一维、有限区间的连续 PDF 与有限整数支持集的离散 PMF，带合法性校验。
        - 自动归一化：PDF/PMF 表达式作为相对权重，系统自动归一化后采样，降低使用门槛。
        - 用户体系：账号注册、登录、退出，数据安全存储于数据库。
        - 三级角色权限体系：访客 / 普通成员 / 管理员。
        - 成员管理：管理员可删除普通成员账号。
        - 操作审计：自动记录自定义分布的新增、修改、删除操作日志。
        - Python 插件：可信开发者可将 `plugin_*.py` 文件放入项目根目录 `extensions/`，重启后自动发现并注册。
        - 统一接口：所有内置分布、自定义分布、外部插件通过同一个仿真服务调用。
        - 结果展示：样本数据表、样本点图、频数图和 XML 下载。
        - 数据存储：支持本地 SQLite 与云端 PostgreSQL 双模式，数据持久化不丢失。
        - 安全机制：数学表达式 AST 白名单解析，不执行任意 Python 代码。
        - 部署支持：可一键部署至 Streamlit Cloud，公网访问无需本地开机。
        """
    )

    st.subheader("角色权限说明")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**👤 访客（未登录）**")
        st.write("- 使用全部仿真功能")
        st.write("- 查看所有内置与自定义分布")
        st.write("- 下载仿真结果 XML")
        st.write("- 不可新增、修改、删除分布")
    with col2:
        st.markdown("**👥 普通成员**")
        st.write("- 访客全部权限")
        st.write("- 新增自定义分布")
        st.write("- 修改、删除自定义分布")
        st.write("- 共同维护分布库")
    with col3:
        st.markdown("**👑 管理员**")
        st.write("- 普通成员全部权限")
        st.write("- 查看所有注册成员列表")
        st.write("- 删除普通成员账号")
        st.write("- 查看全部操作审计日志")

    st.markdown(f"**外部插件目录：** `{EXTENSIONS_DIR}`")
    st.success(f"本次启动自动加载外部 Python 插件：{len(extension_report.loaded_ids)} 个。")
    if extension_report.loaded_ids:
        st.write("已加载：" + "、".join(extension_report.loaded_ids))
    if extension_report.errors:
        st.warning("以下插件未能加载：\n" + "\n".join(extension_report.errors))

    if st.session_state.get("is_logged_in") and st.session_state.get("user_role") == "admin":
        st.divider()
        st.subheader("管理员：成员管理")
        users = user_repo.list_all_users()
        role_map = {"admin": "管理员", "member": "普通成员"}
        table_data = [
            {"用户名": u.username, "角色": role_map.get(u.role, u.role), "注册时间": u.created_at}
            for u in users
        ]
        st.dataframe(table_data, use_container_width=True)

        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            member_list = [u.username for u in users if u.role != "admin"]
            if member_list:
                del_target = st.selectbox("选择要删除的成员", options=member_list)
            else:
                st.caption("暂无普通成员可删除")
                del_target = None
        with col_b:
            confirm_del_user = st.checkbox("确认删除该成员")
        with col_c:
            if del_target and st.button("删除成员", type="secondary", disabled=not confirm_del_user):
                try:
                    user_repo.delete_user(del_target, st.session_state["current_user"])
                    st.success(f"已删除成员：{del_target}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"删除失败：{exc}")

        st.divider()
        st.subheader("管理员：操作审计日志")
        logs = audit_repo.list_recent(limit=100)
        if logs:
            log_table = [
                {"操作时间": log.created_at, "操作人": log.operator, "操作类型": log.action, "操作对象": log.target, "详情": log.detail}
                for log in logs
            ]
            st.dataframe(log_table, use_container_width=True, height=400)
        else:
            st.info("暂无操作记录")

    st.info(
        "当前支持本地运行与云端部署两种模式。Python 插件文件导入时会执行代码，因此 extensions 目录只能由可信开发者维护；"
        "公开部署前还需要完善成员角色权限、服务端资源限制和部署配置。"
    )


def main() -> None:
    st.set_page_config(
        page_title="随机分布仿真 Agent",
        page_icon="📊",
        layout="wide",
    )
    apply_custom_style()
    st.title("随机分布仿真 Agent")
    st.caption("27 种内置分布 · 统一接口 · 自定义扩展 · 样本可视化 · XML 导出")

    init_auth_state()
    repository, user_repo, audit_repo, registry, manager, simulation_service, extension_report = build_runtime()
    custom_definitions = repository.list_all()

    render_sidebar_auth(user_repo)

    workspace_tab, custom_tab, about_tab = st.tabs(
        ["仿真工作台", "自定义分布管理", "系统说明"]
    )
    with workspace_tab:
        render_simulation_workspace(registry, custom_definitions, simulation_service)
    with custom_tab:
        render_custom_distribution_manager(manager, repository)
    with about_tab:
        render_about_page(user_repo, audit_repo, extension_report)


if __name__ == "__main__":
    main()
