# 外部 Python 插件目录

将可信开发者编写的 `plugin_*.py` 文件放在本目录，重启 Agent 后会自动发现并注册。

示例文件位于 `examples/plugin_bernoulli.py`。如需体验：

1. 将该示例复制到本目录；
2. 确认文件名仍以 `plugin_` 开头；
3. 重启 Streamlit；
4. 在仿真工作台的分布列表中找到“示例伯努利分布（外部插件）”。

每个插件文件必须提供 `build_plugin()`，返回实现以下接口的对象：

- `metadata`：含 distribution_id、name、kind、dimension、version、parameters；
- `validate_parameters(params)`：验证并返回规范化参数；
- `sample(n, params, rng)`：返回 n 个样本。

**安全边界：** Python 插件在导入时会执行代码，因此本目录只能由可信开发者/管理员维护。不得把公网用户上传的 Python 文件放入此目录，也不得将其当作普通用户的网页自定义分布功能。
