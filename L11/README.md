# torch_npu 集合通信作业完成报告

## 一、作业目标

在 2 张 Ascend NPU（rank0 / rank1）上完成如下分布式流程：

1. 每张 NPU 卡初始化一个 tensor（rank0 上是 tensor0，rank1 上是 tensor1）；
2. 通过 **AllGather** 让每个 rank 都收集到两个 tensor，并按第 0 维拼接成一个大 tensor；
3. 随机初始化一个参数 **weight**，对拼接后的大 tensor 做 **matmul**；
4. 对 matmul 结果做 **AllReduce**，打印最终结果。

## 二、文件清单（均在当前目录）

| 文件 | 说明 |
|---|---|
| `allgather_matmul_allreduce.py` | 主程序，包含 Step1~Step4 全部逻辑 |
| `run.sh` | 一键启动脚本（torchrun 拉起 2 个进程） |
| `run.log` | 最近一次运行的完整输出日志 |
| `README.md` | 本报告 |

## 三、如何运行

```bash
cd /workspace/torch_homework/11
bash run.sh
```

等价于手动执行：

```bash
torchrun --nproc_per_node=2 allgather_matmul_allreduce.py
```

- `--nproc_per_node=2` 表示本节点启动 2 个进程，分别绑定到 NPU 0 和 NPU 1。
- `torchrun`（PyTorch 1.9+ 标准启动器）会自动为每个进程注入 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`、`MASTER_ADDR`、`MASTER_PORT` 等环境变量，rank0 兼任通信组主节点（master）。
- 若想修改卡数，同时改 `--nproc_per_node` 的值即可（本机已确认 `torch.npu.device_count() == 2`）。

## 四、实现原理

### 1. 分布式进程组与 HCCL

```python
import torch.distributed as dist
import torch_npu  # 关键：导入即向 torch.distributed 注册 "hccl" 后端

dist.init_process_group(backend="hccl", timeout=timedelta(minutes=30))
```

- **HCCL**（Huawei Collective Communication Library）是昇腾 NPU 专用的集合通信库，地位等同于 GPU 上的 NCCL，底层基于 NPU 间的 RDMA / PCIe 链路，实现 AllGather、AllReduce、Broadcast、ReduceScatter 等集合通信原语。
- `import torch_npu` 之后，`hccl` 后端自动注册进 `torch.distributed`（本机验证 `hccl` 已出现在 `dist.Backend.backend_type_map` 中），因此可以直接复用 PyTorch 原生分布式 API，上层代码与 GPU 版几乎一致。
- 每个进程通过 `torch.npu.set_device(local_rank)` 绑定对应的 NPU 卡，之后所有 `device='npu:x'` 的张量计算和通信都发生在该卡上。

### 2. Step1：各卡本地初始化

```python
base = rank * 100 + 1
local_tensor = torch.arange(base, base + 8, dtype=torch.float32, device=device).reshape(2, 4)
```

- rank0 得到 `tensor0 = [[1,2,3,4],[5,6,7,8]]`，rank1 得到 `tensor1 = [[101,...,108]]`。用可区分的固定值（而非随机数）是为了在 AllGather 后能直观验证拼接顺序的正确性。
- 每个进程只持有自己那张卡上的数据，shape 为 `[2, 4]`。

### 3. Step2：AllGather + 按第 0 维拼接

```python
tensor_list = [torch.empty(local_tensor.shape, dtype=..., device=device) for _ in range(world_size)]
dist.all_gather(tensor_list, local_tensor)   # 收集所有 rank 的张量
big_tensor = torch.cat(tensor_list, dim=0)   # 按第 0 维拼接
```

- **AllGather** 是所有进程都参与的集合通信：每个 rank 把自己的 `local_tensor` 广播出去，同时收到所有 rank（包括自己）的张量，结果存放在 `tensor_list` 中，`tensor_list[i]` 即 rank i 的张量。
- 通信后每张卡上都持有相同的数据副本，随后 `torch.cat(..., dim=0)` 按第 0 维拼成 `[4, 4]` 的大 tensor（rank0 的 2 行在前，rank1 的 2 行在后）。
- 这是数据并行/模型并行场景的常用模式：把各卡上的分片汇总成全局完整数据。

### 4. Step3：随机初始化 weight 并做 matmul

```python
torch.npu.manual_seed(1234)  # 所有 rank 相同种子，保证各卡 weight 一致
weight = torch.randn(4, 3, dtype=torch.float32, device=device)
mm_result = torch.matmul(big_tensor, weight)  # [4,4] @ [4,3] -> [4,3]
```

- `torch.npu.manual_seed(1234)` 让所有 rank 生成完全相同的 weight，从而各卡 matmul 结果天然一致（这也让 Step4 的 AllReduce 行为可以被清晰解读）。
- matmul 直接在 NPU 上执行，由 Ascend 硬件矩阵计算单元完成。

### 5. Step4：AllReduce 并打印

```python
dist.all_reduce(mm_result, op=dist.ReduceOp.SUM)
```

- **AllReduce**：对所有 rank 的 tensor 按指定算子（此处为 SUM）归约，再把结果广播回每个 rank，通信后每个 rank 持有相同的最终结果。
- 由于 AllGather 后各卡输入相同、weight 也相同，各卡 matmul 结果本应一致，因此 `AllReduce(SUM)` 的最终值 = 单卡 matmul 结果 × world_size（×2），这是集合通信的正常语义，实测输出符合该规律（如 `-1.8003 → -3.6006`）。
- 两个 rank 均打印最终结果，验证"每个 rank 都收到了相同且正确的归约结果"。

### 6. 数据流总览

```
rank0: tensor0[2,4] ─┐
                     ├─ AllGather ─> 每卡得到 [tensor0; tensor1] = big[4,4]
rank1: tensor1[2,4] ─┘                    │
                                          ├─ matmul(weight[4,3]) ─> mm[4,3]
                                          │
                                          └─ AllReduce(SUM) ─> 每卡得到相同最终结果 [4,3]
```

## 五、运行结果解读（摘自 run.log）

- **Step1**：rank0 输出 `[[1,2,3,4],[5,6,7,8]]`，rank1 输出 `[[101..104],[105..108]]`，各自持有本地张量。
- **Step2**：两卡输出的 `big_tensor` 完全相同且为 `[4, 4]`，第 0~1 行来自 rank0、第 2~3 行来自 rank1，拼接顺序正确。
- **Step3**：两卡 weight 数值一致，matmul 结果一致（如第一行 `[-1.8003, 14.032, -0.4407]`）。
- **Step4**：AllReduce 后结果恰为 Step3 的 2 倍（如 `[-1.8003 → -3.6006]`），且两卡一致，符合 SUM 归约语义。

> 备注：日志中 `Device do not support double dtype` 为 torch_npu 内部提示（无害警告），不影响计算；已通过显式指定 dtype=float32 规避。

## 六、可复现性与扩展

- **可复现**：tensor 用固定值生成，weight 用固定随机种子（1234），多次运行结果一致。
- **扩展方向**：
  - 需要跨机多卡时，将 `torchrun --nproc_per_node=2` 改为 `--nnodes=N --nproc_per_node=M` 并设置 `--rdzv-endpoint`；
  - 若想观察"各卡结果不同"时的 AllReduce，可去掉统一 seed、或让各卡 matmul 后加上 rank 相关偏置；
  - `dist.all_gather` 也可替换为更高效的 `dist.all_gather_into_tensor`（torch 1.13+，单块连续显存/NPU 内存）。
