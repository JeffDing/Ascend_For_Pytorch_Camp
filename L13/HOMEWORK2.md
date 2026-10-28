# 作业 2 完成报告：Stateful 训练状态保存与恢复

## 环境

- 运行方式（`--device` 是脚本自身的参数，必须放在脚本名**之后**；放在 `torchrun` 选项区会报 `unrecognized arguments: --device`）：
  ```bash
  torchrun --standalone --nproc-per-node=2 02_dcp_stateful.py --device cpu
  ```
- 后端：NPU + HCCL（脚本按 `--device npu` 走 `torch_npu.transfer_to_npu()` + `hccl`；实际验证在 CPU + Gloo）
- PyTorch 版本：2.10.0
- torch_npu 版本：2.10.0
- NPU 型号：Ascend 910，npu-smi 26.1.1
- 可用 NPU 设备数：2
- CPU 型号：华为鲲鹏（Kunpeng）ARM64（`/proc/cpuinfo`：implementer 0x48，架构 aarch64）
- 文件：`/workspace/torch_homework/13/02_dcp_stateful.py`

## 代码改动

### `StatefulTrainer.state_dict`（TODO-1）

返回 `{"model": ..., "optim": ...}` 嵌套 dict，让 DCP 知道要保存哪些子状态：

```python
def state_dict(self) -> dict:
    return {"model": self.model.state_dict(), "optim": self.optim.state_dict()}
```

### `StatefulTrainer.load_state_dict`（TODO-2）

按同样结构把嵌套 dict 交还给模型与优化器，二者均为就地修改：

```python
def load_state_dict(self, state_dict: dict) -> None:
    self.model.load_state_dict(state_dict["model"])
    self.optim.load_state_dict(state_dict["optim"])
```

### `save_trainer`（TODO-3）

state_dict 顶层键固定为 `trainer`，DCP 发现值是 `Stateful` 对象后自动调用其 `state_dict()`：

```python
def save_trainer(trainer, ckpt_dir):
    dcp.save({"trainer": trainer}, checkpoint_id=ckpt_dir)
```

### `load_trainer`（TODO-4）

`dcp.load` 先原地填充同构嵌套 dict，再调用 `load_state_dict` 完成回写：

```python
def load_trainer(trainer, ckpt_dir):
    dcp.load({"trainer": trainer}, checkpoint_id=ckpt_dir)
```

## 运行结果与验收

假训练 3 步后保存，新建模型 + 优化器并用 `warm_up_optimizer` 预热出目的地张量后恢复，逐项校验：

| 校验项 | 要求 | 实际结果 |
|---|---|---|
| 模型权重 | 与保存时逐元素一致 | ✅ 每个参数的 `torch.allclose(p.detach(), before["weight"][name])` 通过 |
| `exp_avg` | 与保存时一致 | ✅ `optim.state[p]["exp_avg"]` 逐参数通过 |
| `exp_avg_sq` | 与保存时一致 | ✅ `optim.state[p]["exp_avg_sq"]` 逐参数通过 |
| `step` | 与保存时一致 | ✅ 优化器状态字段完整，数值一致 |
| `param_groups[0]["lr"]` | 非张量字段恢复成功 | ✅ `0.01 == 0.01` |

程序输出：

```
[rank0] 保存完成: 训练状态已写入 ./ckpt_stateful (lr=0.01)
[rank1] 保存完成: 训练状态已写入 ./ckpt_stateful (lr=0.01)
[rank1] 恢复完成: 权重 / exp_avg / step / lr 全部一致
[rank0] 恢复完成: 权重 / exp_avg / step / lr 全部一致
[rank1] PASS: 作业 2 全部断言通过
[rank0] PASS: 作业 2 全部断言通过
```

## 简答题

**`warm_up_optimizer` 为什么不可省略？为什么 DCP 加载要求目的地张量先存在？**

Adam 是懒创建优化器：第一次 `optim.step()` 之前 `optim.state` 为空，`exp_avg` / `exp_avg_sq` / `step` 都不存在。而 `dcp.load` 的语义是**原地填充**——它只会往已有的张量里写数据，不会替目的地创建新张量。因此必须用零梯度 step 先把所有状态张量创建出来，否则加载时报 shape 不匹配或 key 缺失。这也正是"完整续训状态"与"权重级恢复"的区别：后者只关心参数值，前者要求优化器内部状态（含步数、动量、学习率）全部就位，才能从断点继续训练。

## 备注

当前容器内 `torch` 与 `torch_npu` 导入时重复注册 `pinned_reserve_segment_size_mb` 的 allocator hook，C++ 层报 `ValueError` 并 abort，必须前置环境变量 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 才能跑通 CPU/Gloo 通路；`--device npu` 的代码路径已按 HCCL 实现，但因该冲突未实际执行。