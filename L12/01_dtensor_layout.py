# 学生模板：补齐 TODO 后运行。
import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), default="cpu")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.device == "npu":
        import torch_npu  # noqa: F401：注册 NPU 设备和 HCCL 后端
        torch.npu.set_device(local_rank)
    elif args.device == "cuda":
        torch.cuda.set_device(local_rank)
    backend = {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[args.device]
    dist.init_process_group(backend=backend)
    if dist.get_world_size() != 2:
        dist.destroy_process_group()
        raise ValueError("本作业请使用 --nproc-per-node=2 启动")
    device = torch.device("cpu" if args.device == "cpu" else f"{args.device}:{local_rank}")
    mesh = init_device_mesh(args.device, (2,))
    return device, mesh


def show(label, tensor):
    # 按 rank 依次打印，便于阅读；所有 rank 都必须调用此函数。
    for rank in range(dist.get_world_size()):
        if dist.get_rank() == rank:
            print(
                f"rank={rank} {label}: placements={tensor.placements}, "
                f"global_shape={tuple(tensor.shape)}, "
                f"local_shape={tuple(tensor.to_local().shape)}\n"
                f"{tensor.to_local().detach().cpu()}",
                flush=True,
            )
        dist.barrier()


def make_layouts(source, mesh):
    # TODO 1：用 distribute_tensor 创建 Shard(0)、Shard(1)、Replicate 三种分布。
    # 按上述顺序返回三个 DTensor。
    row = distribute_tensor(source, mesh, placements=[Shard(0)])
    col = distribute_tensor(source, mesh, placements=[Shard(1)])
    replicated = distribute_tensor(source, mesh, placements=[Replicate()])
    return row, col, replicated


def restore(sharded):
    # TODO 2：用 redistribute 把 Shard(0) 转为 Replicate。
    # 返回转换后的 DTensor。
    return sharded.redistribute(placements=[Replicate()])


def main():
    device, mesh = setup()
    try:
        source = torch.arange(32, dtype=torch.float32, device=device).reshape(8, 4)
        row, col, replicated = make_layouts(source, mesh)
        rank = dist.get_rank()
        for label, value, placement, shape, expected in (
            ("Shard(0)", row, Shard(0), (4, 4), source.chunk(2, dim=0)[rank]),
            ("Shard(1)", col, Shard(1), (8, 2), source.chunk(2, dim=1)[rank]),
            ("Replicate", replicated, Replicate(), (8, 4), source),
        ):
            assert value.placements == (placement,)
            assert tuple(value.shape) == (8, 4)
            assert tuple(value.to_local().shape) == shape
            torch.testing.assert_close(value.to_local(), expected)
            show(label, value)
        recovered = restore(row)  # 所有 rank 都要参与集合通信。
        assert recovered.placements == (Replicate(),)
        torch.testing.assert_close(recovered.to_local(), source)
        show("恢复后的张量", recovered)
        if rank == 0:
            print("PASS: 作业 1 全部断言通过", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
