# DTensor 课后作业

两道必做题，每题补齐一个小脚本，预计共 40～50 分钟。不需要下载模型或数据集。

## 环境与运行

使用课程提供的 PyTorch 环境，需要支持 `torch.distributed.tensor`。默认使用 CPU + Gloo，在一台电脑上启动两个进程即可，不需要两张加速卡。GPU 使用 CUDA + NCCL，NPU 使用与课程环境配套的 torch_npu + HCCL；这两种方式均需要两张可用设备。

在本目录执行：

```bash
torchrun --standalone --nproc-per-node=2 01_dtensor_layout.py
torchrun --standalone --nproc-per-node=2 02_dtensor_partial.py
```

使用 GPU 或 NPU 时，在命令末尾分别添加 `--device cuda` 或 `--device npu`。

模板已经提供进程组初始化、DeviceMesh、固定数据、打印和验收断言。只需补齐 TODO 函数，删除对应的 `raise NotImplementedError(...)`，无需修改其他代码。未完成时出现该异常是预期行为。

## 作业 1：观察张量的分布方式

文件：`01_dtensor_layout.py`

给定包含 0～31 的 `[8, 4]` 张量，在两个 rank 上完成以下操作：

1. 在 `make_layouts` 中使用 `distribute_tensor`，分别创建 `Shard(0)`、`Shard(1)`、`Replicate()` 的 DTensor，并按此顺序返回。
2. 观察程序打印的各 rank 的 placement、全局 shape、本地 shape 和本地数据。
3. 在 `restore` 中使用 `redistribute`，将按第 0 维分片的张量转换为 `Replicate()`。

验收要求：

| 分布方式 | 全局 shape | 每个 rank 的本地 shape |
|---|---|---|
| `Shard(0)` | `[8, 4]` | `[4, 4]` |
| `Shard(1)` | `[8, 4]` | `[8, 2]` |
| `Replicate()` | `[8, 4]` | `[8, 4]` |

本地数据与对应切片一致，恢复后的数据与原张量一致，最终打印 `PASS: 作业 1 全部断言通过`。

**一句话回答：为什么全局 shape 没变，本地 shape 却变了？**

## 作业 2：理解 Partial 为什么需要归约

文件：`02_dtensor_partial.py`

给定固定矩阵 `X[4, 8]` 和 `W[8, 4]`，普通张量计算的 `X @ W` 作为参考结果。

1. 在 `partial_matmul` 中将 X 按 `Shard(1)` 分布，将 W 按 `Shard(0)` 分布。
2. 直接对两个 DTensor 做矩阵乘法，并返回结果。观察输出的 placement 和本地数据。
3. 在 `finish_sum` 中使用 `redistribute`，将结果转换为 `Replicate()`。

验收要求：

- 矩阵乘法结果为 `Partial()`，每个 rank 本地 shape 均为 `[4, 4]`。
- 各 rank 的本地结果等于对应输入分片的矩阵乘积。
- 转换为 `Replicate()` 后，每个 rank 的结果均与普通张量参考结果一致。
- 最终打印 `PASS: 作业 2 全部断言通过`。

请使用 DTensor 完成运算，不要手动调用 `all_reduce`，也不要直接把参考结果包装成输出。本题的 W 是数学记法中的 `[in, out]` 矩阵，不是 `nn.Linear.weight`。

**一句话回答：为什么每个 rank 的输出 shape 已经完整，仍然需要求和？**

## 提交内容

- 补齐后的两个 Python 脚本。
- 两题运行日志，需包含各 rank 的输出和 PASS 提示。
- 在 `answers.txt` 中写出两道简答题的回答，并注明使用的是 CPU、GPU 还是 NPU。

提示：涉及通信的操作必须由所有 rank 共同执行，不能只放在 `if rank == 0` 分支里。模板中的打印函数也需要所有 rank 调用。
