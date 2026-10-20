# 作业 1 完成报告：观察张量的分布方式

## 环境

- 运行方式：`torchrun --standalone --nproc-per-node=2 <脚本名> --device npu`（`--device` 是脚本自身的参数，必须放在脚本名**之后**；放在 torchrun 选项区会报 `unrecognized arguments: --device`）
- 后端：NPU + HCCL
- PyTorch 版本：2.10.0，torch_npu 版本：2.1.0
- NPU 型号：Ascend 910（`Ascend910_9382`），npu-smi 26.1.1
- 可用 NPU 设备数：2
- CPU 型号：华为鲲鹏（Kunpeng）ARM64（`/proc/cpuinfo`：implementer 0x48，part 0xd02；容器内 `lscpu` 的 Model name 为 unknown）
- 文件：`/workspace/torch_homework/12/01_dtensor_layout.py`

## 代码改动

### `make_layouts`（TODO 1）

使用 `distribute_tensor` 将同一个 `[8, 4]` 源张量按三种 placement 分发到 2 个 rank 的 DeviceMesh 上，并按 `Shard(0)`、`Shard(1)`、`Replicate()` 顺序返回：

```python
def make_layouts(source, mesh):
    row = distribute_tensor(source, mesh, placements=[Shard(0)])
    col = distribute_tensor(source, mesh, placements=[Shard(1)])
    replicated = distribute_tensor(source, mesh, placements=[Replicate()])
    return row, col, replicated
```

### `restore`（TODO 2）

调用 DTensor 的 `redistribute`，把按第 0 维分片的张量转换为 `Replicate()`，等价于一次跨 rank 的 all-gather：

```python
def restore(sharded):
    return sharded.redistribute(placements=[Replicate()])
```

## 运行结果与验收

三种分布的打印与断言均通过，每个 rank 的本地数据与对应切片一致：

| 分布方式 | 全局 shape | rank 0 本地 shape | rank 1 本地 shape |
|---|---|---|---|
| `Shard(0)` | `[8, 4]` | `[4, 4]`（行 0～3） | `[4, 4]`（行 4～7） |
| `Shard(1)` | `[8, 4]` | `[8, 2]`（列 0～1） | `[8, 2]`（列 2～3） |
| `Replicate()` | `[8, 4]` | `[8, 4]` | `[8, 4]` |

`restore(row)` 后两个 rank 的本地张量均恢复为完整的 0～31 原张量，程序最终输出：

```
PASS: 作业 1 全部断言通过
```

## 简答题

**为什么全局 shape 没变，本地 shape 却变了？**

全局 shape 是 DTensor 的逻辑视图，描述的是"完整张量有多大"，它与分布方式无关，只由源张量决定；而本地 shape 是每个 rank 实际持有的物理分片大小，由 placement 在对应维上按 rank 数均分得到，因此 `Shard(0)`、`Shard(1)`、`Replicate()` 三种分布下本地 shape 不同。