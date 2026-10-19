"""
torch_npu 集合通信作业：AllGather -> MatMul -> AllReduce

环境：2 张 Ascend NPU（rank0 / rank1），由 torchrun 拉起 2 个进程。
流程：
  Step 1  每张卡初始化一个 tensor（rank0 -> tensor0，rank1 -> tensor1）
  Step 2  AllGather 收集所有 rank 的 tensor，按第 0 维拼接成大 tensor
  Step 3  随机初始化 weight，对大 tensor 做 matmul
  Step 4  对 matmul 结果做 AllReduce，打印最终结果
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu


def main():
    # ==================== Step 0: 初始化分布式进程组 ====================
    # backend="hccl"：Huawei Collective Communication Library，NPU 专用集合通信库。
    # torchrun 会自动注入 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT / LOCAL_RANK。
    dist.init_process_group(backend="hccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank()                       # 全局 rank：0 / 1
    world_size = dist.get_world_size()           # 卡数：2
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)             # 本进程绑定对应的 NPU 卡
    device = torch.device(f"npu:{local_rank}")
    print(f"[rank {rank}] world_size={world_size}, device={device}")

    # ==================== Step 1: 每张卡初始化一个 tensor ====================
    # 用 arange 生成可区分的固定值，便于验证 AllGather 的正确性：
    #   rank0: [[  1,  2,  3,  4], [  5,  6,  7,  8]]
    #   rank1: [[101,102,103,104], [105,106,107,108]]
    base = rank * 100 + 1
    local_tensor = torch.arange(
        base, base + 8, dtype=torch.float32, device=device
    ).reshape(2, 4)
    print(f"[rank {rank}] Step1 local_tensor (shape={tuple(local_tensor.shape)}) =\n{local_tensor}")

    # ==================== Step 2: AllGather 收集所有 rank 的 tensor，按第 0 维拼接 ====================
    tensor_list = [
        torch.empty(local_tensor.shape, dtype=local_tensor.dtype, device=device)
        for _ in range(world_size)
    ]
    dist.all_gather(tensor_list, local_tensor)   # 每张卡都拿到所有 rank 的 tensor
    big_tensor = torch.cat(tensor_list, dim=0)   # 按第 0 维拼接 -> [4, 4]
    print(f"[rank {rank}] Step2 gathered big_tensor (shape={tuple(big_tensor.shape)}) =\n{big_tensor}")

    # ==================== Step 3: 随机初始化 weight，对 big_tensor 做 matmul ====================
    torch.npu.manual_seed(1234)                  # 所有 rank 使用相同种子 -> 各卡 weight 一致
    weight = torch.randn(4, 3, dtype=torch.float32, device=device)
    mm_result = torch.matmul(big_tensor, weight)  # [4, 4] @ [4, 3] -> [4, 3]
    print(f"[rank {rank}] Step3 weight (shape={tuple(weight.shape)}) =\n{weight}")
    print(f"[rank {rank}] Step3 matmul result (AllReduce 之前) =\n{mm_result}")

    # ==================== Step 4: 对 matmul 结果做 AllReduce（求和） ====================
    # 由于 AllGather 后各卡数据一致、weight 一致，AllReduce(SUM) 的结果 = matmul 结果 * world_size
    dist.all_reduce(mm_result, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] Step4 matmul result (AllReduce 之后，最终结果) =\n{mm_result}")

    # 销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
