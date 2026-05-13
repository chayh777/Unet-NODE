# NODE适配器RK4求解器实现规格

> **Goal:** 在NODE适配器中实现四阶Runge-Kutta (RK4) 求解器，提升数值精度

> **Architecture:** 在 NODEAdapter 中添加 solver 参数（默认 "euler" 保持向后兼容），通过条件分支实现欧拉法和RK4求解器。配置系统保持不变，只需在 YAML 中添加 `solver: rk4`。

> **Tech Stack:** PyTorch, pytest

---

## 背景

当前 NODE 适配器使用显式欧拉法（一阶方法），实现简单但精度较低。RK4 是经典的高阶方法，每步需要 4 次 ODE 函数评估，精度显著提升。

## 功能需求

1. **求解器选择**: 支持 euler 和 rk4 两种求解器，通过配置选择
2. **向后兼容**: 默认使用 euler，现有配置无需修改
3. **参数语义保持**: step_size × steps = 总积分时间 T

## 实现方案

### 1. NODEAdapter 修改

在 `src/models/node_adapter.py` 中：

- 添加 `solver` 参数 (str, 默认 "euler")
- 实现 `_forward_euler()` 和 `_forward_rk4()` 方法
- `forward()` 方法根据 solver 选择调用对应方法

### 2. RK4 算法

```
k1 = f(z)
k2 = f(z + 0.5*h*k1)
k3 = f(z + 0.5*h*k2)
k4 = f(z + h*k3)
z_new = z + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
```

其中 h = step_size

### 3. 配置扩展

在 YAML 配置中添加：
```yaml
node:
  solver: rk4  # 或 "euler"
  steps: 4
  step_size: 0.25
```

### 4. 测试用例

- 测试 RK4 求解器输出形状与欧拉法一致
- 测试 RK4 梯度反向传播正常
- 测试配置解析支持 solver 字段
- (可选) 数值精度对比测试

## 文件变更清单

| 文件 | 变更 |
|------|------|
| `src/models/node_adapter.py` | 添加 solver 参数和 RK4 实现 |
| `tests/test_adapters.py` | 添加 RK4 相关测试 |
| `configs/experiments/*.yaml` | (可选) 添加 solver 字段用于实验 |

## 验收标准

1. 现有使用欧拉法的测试继续通过
2. 新增 RK4 求解器可通过配置启用
3. 梯度反向传播正常工作
4. 输出形状与输入一致