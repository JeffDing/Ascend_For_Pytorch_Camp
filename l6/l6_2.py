import torch
import torch_npu
import gc

# 初始化NPU设备
device = torch.device("npu:0")
torch_npu.npu.init()

def run_npu_op():
    """执行NPU matmul算子，产生算子临时显存分配"""
    # 用户侧创建输入张量
    a = torch.randn(2048, 2048, device=device, dtype=torch.float32)
    b = torch.randn(2048, 2048, device=device, dtype=torch.float32)
    # NPU matmul算子，算子内部会申请临时workspace显存
    c = torch.matmul(a, b)
    # 同步流，确保算子执行完成、内存分配动作被完整记录
    torch_npu.npu.synchronize()
    return a, b, c

if __name__ == "__main__":
    # 清理历史显存、GC，消除干扰
    gc.collect()
    torch_npu.npu.empty_cache()
    torch_npu.npu.reset_peak_memory_stats(device)

    # 1. 开启NPU内存历史记录（参数max_entries可放大记录容量）
    torch.npu.memory._record_memory_history(max_entries=100000)

    # 2. 执行NPU算子逻辑
    tensor_a, tensor_b, tensor_c = run_npu_op()

    # 3. 导出内存快照文件
    torch.npu.memory._dump_snapshot("npu_op_memory_snapshot.pickle")
    print("内存快照已导出: npu_op_memory_snapshot.pickle")

    # 关闭内存记录，释放埋点开销
    torch.npu.memory._record_memory_history(None)