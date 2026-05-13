# NODE适配器RK4求解器实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在NODE适配器中实现四阶Runge-Kutta (RK4) 求解器，通过配置选择求解器类型

**Architecture:** 在 NODEAdapter 中添加 `solver` 参数，默认 "euler" 保持向后兼容。实现条件分支选择欧拉法或RK4求解器

**Tech Stack:** PyTorch, pytest

---

## 文件结构

- 修改: `src/models/node_adapter.py` (添加 solver 参数和 RK4 实现)
- 修改: `src/models/segmentation_model.py` (传递 solver 参数)
- 修改: `src/experiments/low_data_runner.py` (配置解析添加 solver)
- 修改: `src/analysis/low_data_geometry.py` (配置解析添加 solver)
- 修改: `src/analysis/robustness_metrics.py` (配置解析添加 solver)
- 修改: `src/analysis/segmentation_compare.py` (配置解析添加 solver)
- 修改: `tests/test_adapters.py` (添加 RK4 测试)

---

## 实施任务

### Task 1: 添加 RK4 相关的失败测试

**Files:**
- Modify: `tests/test_adapters.py`
- Run: `pytest tests/test_adapters.py::test_node_adapter_rk4_preserves_shape -v`

- [ ] **Step 1: 添加 RK4 测试用例**

在 `tests/test_adapters.py` 末尾添加:

```python
def test_node_adapter_rk4_preserves_shape():
    x = torch.randn(2, 32, 8, 8)
    y = NODEAdapter(
        channels=32,
        hidden_channels=32,
        steps=4,
        step_size=0.25,
        solver="rk4",
    )(x)
    assert tuple(y.shape) == (2, 32, 8, 8)


def test_node_adapter_rk4_backward_smoke():
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    model = NODEAdapter(
        channels=32,
        hidden_channels=32,
        steps=2,
        step_size=0.25,
        solver="rk4",
    )
    y = model(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert any(p.grad is not None for p in model.parameters())


def test_node_adapter_euler_default_solver():
    x = torch.randn(2, 32, 8, 8)
    model = NODEAdapter(
        channels=32,
        hidden_channels=32,
        steps=4,
        step_size=0.25,
    )
    y = model(x)
    assert tuple(y.shape) == (2, 32, 8, 8)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_adapters.py::test_node_adapter_rk4_preserves_shape -v`

Expected: FAIL with "unexpected keyword argument 'solver'"

- [ ] **Step 3: Commit**

```bash
git add tests/test_adapters.py
git commit -m "test: add RK4 solver tests (failing)"
```

---

### Task 2: 实现 RK4 求解器

**Files:**
- Modify: `src/models/node_adapter.py:25-48`

- [ ] **Step 1: 修改 NODEAdapter 类添加 solver 参数**

编辑 `src/models/node_adapter.py`，修改 `NODEAdapter.__init__`:

```python
class NODEAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        steps: int,
        step_size: float,
        init: AdapterInit = "default",
        solver: str = "euler",
    ) -> None:
        super().__init__()
        if solver not in ("euler", "rk4"):
            raise ValueError(f"solver must be 'euler' or 'rk4', got {solver!r}")
        self.func = ODEFunction(
            channels=channels,
            hidden_channels=hidden_channels,
            init=init,
        )
        self.steps = steps
        self.step_size = step_size
        self.solver = solver
```

- [ ] **Step 2: 修改 forward 方法实现 RK4**

编辑 `src/models/node_adapter.py`，修改 `NODEAdapter.forward`:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.solver == "rk4":
            return self._forward_rk4(x)
        return self._forward_euler(x)

    def _forward_euler(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for _ in range(self.steps):
            z = z + self.step_size * self.func(z)
        return z

    def _forward_rk4(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        h = self.step_size
        for _ in range(self.steps):
            k1 = self.func(z)
            k2 = self.func(z + 0.5 * h * k1)
            k3 = self.func(z + 0.5 * h * k2)
            k4 = self.func(z + h * k3)
            z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z
```

- [ ] **Step 3: 运行测试验证通过**

Run: `pytest tests/test_adapters.py::test_node_adapter_rk4_preserves_shape tests/test_adapters.py::test_node_adapter_rk4_backward_smoke tests/test_adapters.py::test_node_adapter_euler_default_solver -v`

Expected: PASS

- [ ] **Step 4: 运行全部适配器测试**

Run: `pytest tests/test_adapters.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/node_adapter.py
git commit -m "feat: add RK4 solver to NODEAdapter"
```

---

### Task 3: 传递 solver 参数到模型构建

**Files:**
- Modify: `src/models/segmentation_model.py:130-171`

- [ ] **Step 1: 在 SegmentationModel 中添加 solver 参数**

编辑 `src/models/segmentation_model.py`，修改 `SegmentationModel.__init__`:

在 `node_step_size: float,` 后添加:
```python
        node_solver: str = "euler",
```

修改 NODEAdapter 构造调用 (line 165-171):
```python
        elif adapter_type == "node":
            self.adapter = NODEAdapter(
                channels=bottleneck_channels,
                hidden_channels=adapter_hidden_channels,
                steps=node_steps,
                step_size=node_step_size,
                init=adapter_init,
                solver=node_solver,
            )
```

- [ ] **Step 2: 更新 build_segmentation_model 函数**

在 `build_segmentation_model` 函数签名中添加 `node_solver` 参数并传递给 SegmentationModel

- [ ] **Step 3: 运行测试验证**

Run: `pytest tests/test_adapters.py -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/models/segmentation_model.py
git commit -m "feat: pass solver param to NODEAdapter in SegmentationModel"
```

---

### Task 4: 配置解析添加 solver 字段

需要修改以下文件，添加对 config["node"]["solver"] 的读取：

1. `src/experiments/low_data_runner.py`
2. `src/analysis/low_data_geometry.py`
3. `src/analysis/robustness_metrics.py`
4. `src/analysis/segmentation_compare.py`

每个文件中：
- 在 `_require_keys(node, ...)` 添加 "solver"
- 在 `build_segmentation_model(...)` 调用中添加 `node_solver=config["node"].get("solver", "euler")`

**Files:**
- Modify: `src/experiments/low_data_runner.py:204-205,326-327`
- Modify: `src/analysis/low_data_geometry.py:121-122,277-278`
- Modify: `src/analysis/robustness_metrics.py:189-190`
- Modify: `src/analysis/segmentation_compare.py:49-50`

- [ ] **Step 1: 修改 low_data_runner.py**

编辑 `src/experiments/low_data_runner.py`:

Line 204-205 改为:
```python
    _require_keys(node, ["steps", "step_size", "solver"], "config.node")
```

Line 326-327 改为:
```python
        node_steps=int(config["node"]["steps"]),
        node_step_size=float(config["node"]["step_size"]),
        node_solver=str(config["node"].get("solver", "euler")),
```

- [ ] **Step 2: 修改 low_data_geometry.py**

Line 121-122 改为:
```python
    _require_keys(node, ["steps", "step_size", "solver"], "config.node")
```

Line 277-278 改为:
```python
        node_steps=int(config["node"]["steps"]),
        node_step_size=float(config["node"]["step_size"]),
        node_solver=str(config["node"].get("solver", "euler")),
```

- [ ] **Step 3: 修改 robustness_metrics.py**

Line 189-190 改为:
```python
            node_steps=int(config["node"]["steps"]),
            node_step_size=float(config["node"]["step_size"]),
            node_solver=str(config["node"].get("solver", "euler")),
```

- [ ] **Step 4: 修改 segmentation_compare.py**

Line 49-50 改为:
```python
        node_steps=int(config["node"]["steps"]),
        node_step_size=float(config["node"]["step_size"]),
        node_solver=str(config["node"].get("solver", "euler")),
```

- [ ] **Step 5: Commit**

```bash
git add src/experiments/low_data_runner.py src/analysis/low_data_geometry.py src/analysis/robustness_metrics.py src/analysis/segmentation_compare.py
git commit -m "feat: add solver field to config parsing"
```

---

## 验收确认

1. `pytest tests/test_adapters.py -v` 全部通过
2. 配置中可指定 `node.solver: rk4` 启用 RK4
3. 现有配置（无 solver 字段）默认使用 euler 向后兼容