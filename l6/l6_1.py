import torch
import torch_npu
import gc

# 指定NPU设备
device = torch.device("npu:0")
torch_npu.npu.init()

def print_npu_memory(stage: str):
    print(f"========== {stage} ==========")
    # 当前张量实际占用显存
    alloc = torch_npu.npu.memory_allocated(device)
    # 内存池向驱动申请的总显存(使用+缓存空闲)
    reserv = torch_npu.npu.memory_reserved(device)
    # 历史峰值实际张量占用
    max_alloc = torch_npu.npu.max_memory_allocated(device)
    # 历史峰值内存池预留大小
    max_reserv = torch_npu.npu.max_memory_reserved(device)
    
    print(f"memory_allocated:     {alloc} Bytes")
    print(f"memory_reserved:      {reserv} Bytes")
    print(f"max_memory_allocated: {max_alloc} Bytes")
    print(f"max_memory_reserved:  {max_reserv} Bytes\n")

# 阶段1：无任何NPU张量
print_npu_memory("1. 未创建NPU Tensor")

# 阶段2：创建大float32张量 2048*2048
x = torch.randn((2048, 2048), device=device)
print_npu_memory("2. 创建大NPU Tensor后")

# 阶段3：删除引用+垃圾回收，重置峰值统计
del x
gc.collect()
torch_npu.npu.reset_peak_memory_stats(device)
print_npu_memory("3. del + gc.collect() 销毁Tensor并重置峰值")

# 阶段4：新建小张量 1024*1024
y = torch.randn((1024, 1024), device=device)
print_npu_memory("4. 复用内存池创建小Tensor")

# 阶段5：销毁小张量，重置峰值
del y
gc.collect()
torch_npu.npu.reset_peak_memory_stats(device)
print_npu_memory("5. 销毁小Tensor完成")
