import gc
import torch
import torch_npu

device = torch.device("npu:0")
torch_npu.npu.init()

# 两条独立NPU流
alloc_stream = torch.npu.Stream(device)
compute_stream = torch.npu.Stream(device)

# 1. 小张量 small pool
with torch.npu.stream(alloc_stream):
    small_t = torch.randn((128, 128), device=device, dtype=torch.float32)
    # 跨流标记，触发Event记录埋点
    small_t.record_stream(compute_stream)

with torch.npu.stream(compute_stream):
    out1 = small_t + 1.0
torch_npu.npu.synchronize(compute_stream)

# 2. 大量 large pool
with torch.npu.stream(alloc_stream):
    large_t = torch.randn((2048, 2048), device=device, dtype=torch.float32)
    large_t.record_stream(compute_stream)

with torch.npu.stream(compute_stream):
    out2 = torch.matmul(large_t, large_t.t())
torch_npu.npu.synchronize(compute_stream)

# 销毁+GC+清空缓存，触发内存池释放埋点
del small_t, large_t, out1, out2
gc.collect()
torch_npu.npu.synchronize()
torch_npu.npu.empty_cache()
print("脚本执行完毕，请查看上方TORCH_NPU memory日志")