# 作业 2 完成报告：理解 Partial 为什么需要归约

## 环境

- 运行方式：`torchrun --standalone --nproc-per-node=2 --device npu`
- 后端：NPU + HCCL
- PyTorch 版本：2.10.0，torch_npu 版本：2.1.0
- NPU 型号：Ascend 910（`Ascend910_9382`），npu-smi 26.1.1
- 可用 NPU 设备数：2
- CPU 型号：华为鲲鹏（Kunpeng）ARM64（`/proc/cpuinfo`：implementer 0x48，part 0xd02；容器内 `lscpu` 的 Model name 为 unknown）
- 文件：`/workspace/torch_homework/12/02_dtensor_partial.py`

## 代码改动

### `partial_matmul`（TODO 1 + TODO 2）

X 按 `Shard(1)`（按列）、W 按 `Shard(0)`（按行）分发到 2 个 rank，然后直接对两个 DTensor 做矩阵乘法。DTensor 自动识别 `Shard(1) @ Shard(0)` 模式，将结果标记为 `Partial(sum)`：

```python
def partial_matmul(x, w, mesh):
    # TODO 1：X 使用 Shard(1)，W 使用 Shard(0)，分别创建 DTensor。
    x_dt = distribute_tensor(x, mesh, placements=[Shard(1)])
    w_dt = distribute_tensor(w, mesh, placements=[Shard(0)])
    # TODO 2：直接对两个 DTensor 做矩阵乘法，返回结果。
    return x_dt @ w_dt
```

### `finish_sum`（TODO 3）

调用 `redistribute(placements=[Replicate()])`，由 DTensor 内部完成跨 rank 的 all-gather + 求和，不手动调用 `all_reduce`：

```python
def finish_sum(partial):
    # TODO 3：用 redistribute 将 Partial 结果转换为 Replicate。
    # 返回转换后的 DTensor，不要手动调用 all_reduce。
    return partial.redistribute(placements=[Replicate()])
```

## 运行结果与验收

| 阶段 | placement | 全局 shape | rank 0 本地 shape | rank 1 本地 shape |
|---|---|---|---|---|
| 归约前的部分和 | `Partial(sum)` | `[4, 4]` | `[4, 4]` | `[4, 4]` |
| 归约后的完整结果 | `Replicate()` | `[4, 4]` | `[4, 4]` | `[4, 4]` |

- rank 0 本地结果 = `x.chunk(2, dim=1)[0] @ w.chunk(2, dim=0)[0]`
- rank 1 本地结果 = `x.chunk(2, dim=1)[1] @ w.chunk(2, dim=0)[1]`
- 两个 rank 归约后均等于普通张量参考值 `x @ w`：

```
[[ 560.,  588.,  616.,  644.],
 [1456., 1548., 1640., 1732.],
 [2352., 2508., 2664., 2820.],
 [3248., 3468., 3688., 3908.]]
```

程序最终输出 `PASS: 作业 2 全部断言通过`。程序中所有通信操作（`distribute_tensor`、`redistribute`、`show` 内的 barrier）均由两个 rank 共同执行，未放入 `if rank == 0` 分支。

## 简答题

**为什么每个 rank 的输出 shape 已经完整，仍然需要求和？**

X 按列分片、W 按行分片后，每个 rank 只算出了完整乘积的一项部分贡献，所有 rank 的结果对应不同的累加项；`Partial` 语义要求把这些部分和跨 rank 求和才能得到真正的 `X @ W`，shape 完整不等于数值完整。