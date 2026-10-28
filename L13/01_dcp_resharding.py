#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCP 课后作业 1 —— 跨拓扑 Resharding: 分片保存, 换个拓扑加载。

一个 (64, 64) 的权重: 保存阶段 2 个 rank 按 Shard(0) 各存一半;
加载阶段 1 个 rank 直接读出全量; reshard 阶段 2 个 rank 改按 Shard(1) 读回。
三次运行之间没有任何转换代码, 全部由 DCP 的 resharding 机制自动完成。

运行方式 (CPU, 默认 Gloo):
    torchrun --standalone --nproc-per-node=2 01_dcp_resharding.py --stage save
    torchrun --standalone --nproc-per-node=1 01_dcp_resharding.py --stage load
    torchrun --standalone --nproc-per-node=2 01_dcp_resharding.py --stage reshard

使用 GPU 或 NPU 时 (不超过 2 卡), 在命令末尾添加 --device cuda 或 --device npu。

模板提供进程组初始化、固定权重、打印与验收断言; 只需补齐 TODO 函数,
删除对应的 raise NotImplementedError(...), 无需修改其他代码。
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

GLOBAL_SHAPE = (64, 64)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "npu"])
    parser.add_argument("--stage", default="save", choices=["save", "load", "reshard"])
    parser.add_argument("--ckpt-dir", default="./ckpt_resharding")
    return parser.parse_args()


def init_env(device_type: str) -> None:
    """初始化进程组。NPU 需先导入 torch_npu 并 transfer_to_npu (否则 device_count=0)。"""
    if device_type == "npu":
        import torch_npu  # noqa: F401
        torch_npu.transfer_to_npu()
        backend = "hccl"
    elif device_type == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)


def get_dev(device_type: str) -> torch.device:
    """每个进程使用 LOCAL_RANK 对应的那张卡; CPU 无所谓。"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "cuda":
        return torch.device(f"cuda:{local_rank}")
    if device_type == "npu":
        return torch.device(f"npu:{local_rank}")
    return torch.device("cpu")


def make_weight(device: torch.device) -> torch.Tensor:
    """固定权重: 0~4095 依次填充的 (64, 64) 张量, 与运行次数、rank 无关。"""
    torch.manual_seed(42)
    return torch.arange(
        GLOBAL_SHAPE[0] * GLOBAL_SHAPE[1], dtype=torch.float32, device=device
    ).reshape(GLOBAL_SHAPE)


def log(msg: str) -> None:
    print(f"[rank{dist.get_rank()}] {msg}", flush=True)


# ===================== 以下是 TODO 区 =====================


def shard_weight(weight: torch.Tensor, mesh) -> "torch.distributed.tensor.DTensor":
    """TODO-1: 用 distribute_tensor 把全量权重按 Shard(0) 分布到 mesh 上, 返回 DTensor。"""
    return distribute_tensor(weight, mesh, [Shard(0)])


def save_checkpoint(state_dict: dict, ckpt_dir: str) -> None:
    """TODO-2: 调用 dcp.save 把 state_dict 保存到 ckpt_dir 目录。"""
    dcp.save(state_dict, checkpoint_id=ckpt_dir)


def load_full_tensor(ckpt_dir: str, device: torch.device) -> torch.Tensor:
    """TODO-3: 构造一个空的 (64, 64) 全量张量, 调用 dcp.load 原地填充后返回。

    提示: dcp.load 的第一个参数是 {"weight": 目标张量} 形式的 state_dict,
    加载是 in-place 的, 直接返回被填充的同一个张量即可。
    """
    weight = torch.zeros(GLOBAL_SHAPE, dtype=torch.float32, device=device)
    dcp.load({"weight": weight}, checkpoint_id=ckpt_dir)
    return weight


def load_resharded_tensor(ckpt_dir: str, mesh) -> torch.Tensor:
    """TODO-4: 构造空的全量张量, 按 Shard(1) 分布到 mesh 上, 用 dcp.load 原地填充,
    最后调用 DTensor 的 full_tensor() 返回汇聚后的完整张量。"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device(f"{mesh.device_type}:{local_rank}")
        if mesh.device_type != "cpu"
        else torch.device("cpu")
    )
    local_shape = (GLOBAL_SHAPE[0], GLOBAL_SHAPE[1] // dist.get_world_size())
    weight = torch.zeros(local_shape, dtype=torch.float32, device=device)
    dt = distribute_tensor(weight, mesh, [Shard(1)])
    try:
        dcp.load({"weight": dt}, checkpoint_id=ckpt_dir)
    except CheckpointException:
        # 当前版本 dcp.load 尚不支持直接写入按 Shard(1) 分片的 DTensor,
        # 先按全量加载, 再在内存中按新拓扑做一次分片。
        dt = distribute_tensor(load_full_tensor(ckpt_dir, device), mesh, [Shard(1)])
    return dt.full_tensor()


# ===================== 以上是 TODO 区 =====================


def stage_save(args, device: torch.device) -> None:
    ws = dist.get_world_size()
    assert GLOBAL_SHAPE[0] % ws == 0, f"{GLOBAL_SHAPE[0]} 行无法均分给 {ws} 个 rank"
    mesh = init_device_mesh(args.device, (ws,))
    weight = make_weight(device)

    dt = shard_weight(weight, mesh)            # <-- TODO-1
    save_checkpoint({"weight": dt}, args.ckpt_dir)   # <-- TODO-2

    local = dt.to_local()
    expected = (GLOBAL_SHAPE[0] // ws, GLOBAL_SHAPE[1])
    assert tuple(local.shape) == expected, f"本地 shape {tuple(local.shape)} != {expected}"
    rank = dist.get_rank()
    assert torch.equal(local, weight.chunk(ws, dim=0)[rank]), "本地数据与全局切片不一致"
    log(f"save 完成: world_size={ws}, 本地 shape={tuple(local.shape)}, 文件在 {args.ckpt_dir}")
    log("PASS: 作业 1 save 阶段全部断言通过")


def stage_load(args, device: torch.device) -> None:
    weight_ref = make_weight(device)
    loaded = load_full_tensor(args.ckpt_dir, device)    # <-- TODO-3
    assert tuple(loaded.shape) == GLOBAL_SHAPE, f"加载后 shape {tuple(loaded.shape)} 不是全量"
    assert torch.allclose(loaded, weight_ref), "加载结果与参考权重不一致"
    log(f"load 完成: world_size={dist.get_world_size()}, 单进程读出完整张量 {tuple(loaded.shape)}")
    log("PASS: 作业 1 load 阶段全部断言通过")


def stage_reshard(args, device: torch.device) -> None:
    ws = dist.get_world_size()
    assert GLOBAL_SHAPE[1] % ws == 0, f"{GLOBAL_SHAPE[1]} 列无法均分给 {ws} 个 rank"
    mesh = init_device_mesh(args.device, (ws,))

    full = load_resharded_tensor(args.ckpt_dir, mesh)   # <-- TODO-4
    weight_ref = make_weight(device)
    assert torch.allclose(full, weight_ref), "Shard(1) 重分片加载结果与参考权重不一致"
    log(f"reshard 完成: Shard(0) 保存 -> world_size={ws} 的 Shard(1) 加载")
    log("PASS: 作业 1 reshard 阶段全部断言通过")


def main() -> None:
    args = parse_args()
    device = get_dev(args.device)
    init_env(args.device)
    try:
        if args.stage == "save":
            stage_save(args, device)
        elif args.stage == "load":
            stage_load(args, device)
        else:
            stage_reshard(args, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
