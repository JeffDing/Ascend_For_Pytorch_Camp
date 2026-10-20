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


def partial_matmul(x, w, mesh):
    # TODO 1：X 使用 Shard(1)，W 使用 Shard(0)，分别创建 DTensor。
    x_dt = distribute_tensor(x, mesh, placements=[Shard(1)])
    w_dt = distribute_tensor(w, mesh, placements=[Shard(0)])
    # TODO 2：直接对两个 DTensor 做矩阵乘法，返回结果。
    return x_dt @ w_dt


def finish_sum(partial):
    # TODO 3：用 redistribute 将 Partial 结果转换为 Replicate。
    # 返回转换后的 DTensor，不要手动调用 all_reduce。
    return partial.redistribute(placements=[Replicate()])


def main():
    device, mesh = setup()
    try:
        x = torch.arange(32, dtype=torch.float32, device=device).reshape(4, 8)
        w = torch.arange(32, dtype=torch.float32, device=device).reshape(8, 4)
        reference = x @ w
        partial = partial_matmul(x, w, mesh)
        assert partial.placements == (Partial(),)
        assert tuple(partial.shape) == (4, 4)
        assert tuple(partial.to_local().shape) == (4, 4)
        rank = dist.get_rank()
        expected_local = x.chunk(2, dim=1)[rank] @ w.chunk(2, dim=0)[rank]
        torch.testing.assert_close(partial.to_local(), expected_local)
        show("归约前的部分和", partial)
        result = finish_sum(partial)  # 所有 rank 都要执行，不能只放在 rank 0 分支里。
        assert result.placements == (Replicate(),)
        torch.testing.assert_close(result.to_local(), reference)
        show("归约后的完整结果", result)
        if rank == 0:
            print("PASS: 作业 2 全部断言通过", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
