# DCP 课后作业

两道必做题，每题补齐一个小脚本，预计共 40～50 分钟。不需要下载模型或数据集。

## 环境与运行

使用课程提供的 PyTorch 环境，需要支持 `torch.distributed.checkpoint`。默认使用 CPU + Gloo，在一台电脑上启动两个进程即可，不需要两张加速卡。GPU 使用 CUDA + NCCL，NPU 使用与课程环境配套的 torch_npu + HCCL；这两种方式均需要两张可用设备。

在本目录执行：

```bash
torchrun --standalone --nproc-per-node=2 01_dcp_resharding.py --stage save
torchrun --standalone --nproc-per-node=1 01_dcp_resharding.py --stage load
torchrun --standalone --nproc-per-node=2 01_dcp_resharding.py --stage reshard
torchrun --standalone --nproc-per-node=2 02_dcp_stateful.py
```

使用 GPU 或 NPU 时，在命令末尾分别添加 `--device cuda` 或 `--device npu`。

> 注意：`--stage` / `--device` 是**脚本自身的参数**，必须放在脚本名**之后**；放在 `torchrun` 选项区会报 `unrecognized arguments`。

模板已经提供进程组初始化、固定数据、打印和验收断言。只需补齐 TODO 函数，删除对应的 `raise NotImplementedError(...)`，无需修改其他代码。未完成时出现该异常是预期行为。

## 作业 1：跨拓扑 Resharding

文件：`01_dcp_resharding.py`

给定一个固定的 `(64, 64)` 权重矩阵，分三个阶段处理，三次运行之间没有任何转换代码：

1. `--stage save`：2 个 rank 用 `distribute_tensor` 把权重按 `Shard(0)` 各存一半，调用 `dcp.save` 保存。
2. `--stage load`：只用 1 个进程，构造一个空的 `(64, 64)` 普通张量，调用 `dcp.load` 原地填充，读出全量。
3. `--stage reshard`：2 个 rank 把空张量按 `Shard(1)` 分布后 `dcp.load`，再用 `full_tensor()` 汇聚校验。

验收要求：

| 阶段 | world_size | 验证内容 |
|---|---|---|
| save | 2 | 本地 shape 为 `(32, 64)`，本地数据与全局第 0 维切片一致，打印 `PASS: 作业 1 save 阶段全部断言通过` |
| load | 1 | 加载结果为完整 `(64, 64)`，与参考权重逐元素一致，打印 `PASS: 作业 1 load 阶段全部断言通过` |
| reshard | 2 | `Shard(0)` 保存的分片按 `Shard(1)` 读回后与参考权重一致，打印 `PASS: 作业 1 reshard 阶段全部断言通过` |

**（选答）思考题：保存拓扑与加载拓扑不同，DCP 为什么能自动完成分片映射？**

## 作业 2：Stateful 训练状态保存与恢复

文件：`02_dcp_stateful.py`

模拟一个最小训练器：两层 MLP + Adam，固定输入假训练 3 步后保存；再新建一份模型和优化器，恢复后逐项校验。需要完成：

1. 实现 `StatefulTrainer.state_dict()`：返回 `{"model": ..., "optim": ...}` 嵌套 state_dict。
2. 实现 `StatefulTrainer.load_state_dict()`：按同样的结构恢复模型与优化器。
3. 实现 `save_trainer` / `load_trainer`：调用 `dcp.save` / `dcp.load`，state_dict 的键写成 `{"trainer": trainer}`。

验收要求：

- 恢复后模型权重与保存时逐元素一致。
- Adam 的 `exp_avg`、`exp_avg_sq`、`step` 与保存时一致（优化器动量不丢，这就是"完整续训状态"）。
- `param_groups` 中的学习率等非张量字段也恢复成功。

**（选答）思考题：为什么必须先调用 `warm_up_optimizer`，才能 `dcp.load` 恢复优化器状态？**

## 提交内容

- 补齐后的两个 Python 脚本。
- 两题运行日志，需包含各 rank 的输出和 PASS 提示。
- 在 `HOMEWORK1.md` / `HOMEWORK2.md` 中记录实现、代码改动、运行环境与验收结果。

提示：涉及通信的操作必须由所有 rank 共同执行，不能只放在 `if rank == 0` 分支里。模板中的打印函数也需要所有 rank 调用。