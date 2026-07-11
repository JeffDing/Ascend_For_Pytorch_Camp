# Ascend for Pytroch第六课

# 环境准备

## 安装torch\-npu

```Bash
pip install torch==2.7.1 torchvision torch-npu==2.7.1
```

## 设置MAX\_JOBS

```Bash
export MAX_JOBS=100
```

# 实践部分

## 题目一：**编写用例，观察****`npu`****类型****`Tensor`****对象创建和销毁前后以下接口执行结果的变化，并对执行结果给出简要的解释。**

```Python
torch.npu.memory_allocated(device=None)
torch.npu.memory_reserved(device=None)
torch.npu.max_memory_allocated(device=None)
torch.npu.max_memory_reserved(device=None)
```

**输出**：

1\. 用例代码；

2\. 执行结果；

3\. 执行结果的简要解释。

### 用例代码`l6_1.py`

```Python
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
```

### 执行结果的简要解释

#### 四个 torch\_npu 内存接口含义

1. torch\_npu\.npu\.memory\_allocated\(device\) 当前时刻所有 NPU Tensor 数据真实占用的显存，只统计正在使用的张量，内存池空闲缓存不计入。

- 创建 Tensor 数值上涨；销毁 GC 后归零；

- reserved 不变代表内存池缓存还在，只是没有张量占用。

2. torch\_npu\.npu\.memory\_reserved\(device\) 昇腾内存池向 NPU 驱动一次性申请的总显存，包含「正在使用的张量内存 \+ 缓存空闲内存块」。

- 首次创建大张量时内存池扩容，reserved 暴涨；

- Tensor 销毁后不会归还硬件，reserved 保持不变；

- 后续新建张量直接复用缓存，不会再次扩容。

3. torch\_npu\.npu\.max\_memory\_allocated\(device\) 上一次调用 `reset_peak_memory_stats` 之后，`memory_allocated` 出现过的最大值，用于定位显存峰值。

- 仅新建张量时刷新；销毁张量不会自动下降，必须手动重置清零。

4. torch\_npu\.npu\.max\_memory\_reserved\(device\) 上一次重置后，内存池预留总显存的历史最大值，代表内存池扩容到的最大规模。

- 内存池扩容时更新；缓存不释放则峰值永久保留，重置后清零。

#### 分阶段现象说明

1. 无张量阶段：无显存分配，四项指标全部为 0。

2. 创建大张量：allocated 等于张量真实大小；内存池扩容导致 reserved 更大；两个 max 更新为当前值，记录峰值。

3. 销毁 \+ GC \+ 重置峰值：张量内存释放回内存池，allocated 归零；内存池不释放显存，reserved 不变；重置后峰值统计清零。

4. 新建小张量：复用内存池空闲块，reserved 不变；allocated 小幅上涨；max\_allocated 更新为小张量尺寸。

5. 销毁小张量：allocated 再次归零，内存池缓存持续保留，重置后峰值清零。

#### 核心特性（昇腾 torch\_npu 特有）

1. 内存池缓存机制：频繁申请释放张量不会反复和驱动交互，提升训练速度；

2. allocated 反映业务实际显存开销，reserved 反映硬件侧真实占用；

3. max 系列接口用于排查 OOM、显存峰值，分段统计必须搭配 `reset_peak_memory_stats`；

4. `torch_npu.npu.*` 是昇腾官方标准 API，生产环境优先使用，`torch.npu` 为上层兼容封装。

## **题目二：编写用例，执行****`NPU`****算子，在内存快照上找到该算子分配内存的调用栈。**

- 内存快照的开启方式参考：

```Python
torch.npu.memory._record_memory_history()
run_your_code()
torch.npu.memory._dump_snapshot("snapshot.pickle")
```

- 内存快照文件可使用 PyTorch在线工具（`https://pytorch.org/memory_viz`）打开分析。

**输出**：1\. 用例代码；2\. 内存快照上的调用栈截图。

### 完整用例代码`l6-2.py`

```Python
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
```


## **题目三：编写用例，构造跨流使用内存的场景，并正确使用****`Tensor.record_stream()`****，根据memory模块的日志确认使用的大小池和****`Event Record`****信息。**



- memory模块日志的开启方法：`export TORCH_NPU_LOGS=+memory`。

    

- 大小池关键字：`using small pool`或`using large pool`。

    

- `Event Record`信息关键字：`Event: record DeviceAllocator is successfully executed`。

    

**输出**：1\. 用例代码；2\. `small/large pool`和`Event Record`相关日志。

### 用例代码`l6_3.py`

```Python
import os
import gc
import torch
import torch_npu

device = torch.device("npu:0")
torch_npu.npu.init()

# 创建两条独立NPU流
alloc_stream = torch.npu.Stream(device)
compute_stream = torch.npu.Stream(device)

# ========== 场景1：小张量 <1MB，命中 small pool ==========
with torch.npu.stream(alloc_stream):
    # 128*128 float32 = 65536 Byte < 1MB
    small_t = torch.randn((128, 128), device=device, dtype=torch.float32)
    # 标记跨流占用，生成Record Event日志
    small_t.record_stream(compute_stream)

# 另一条流使用张量
with torch.npu.stream(compute_stream):
    out1 = small_t + 1.0
torch_npu.npu.synchronize(compute_stream)

# ========== 场景2：大量 >1MB，命中 large pool ==========
with torch.npu.stream(alloc_stream):
    # 2048*2048 float32 = 16777216 Byte > 1MB
    large_t = torch.randn((2048, 2048), device=device, dtype=torch.float32)
    large_t.record_stream(compute_stream)

with torch.npu.stream(compute_stream):
    out2 = torch.matmul(large_t, large_t.t())
torch_npu.npu.synchronize(compute_stream)

# 主动销毁引用、GC、清空缓存，触发内存池释放校验日志
del small_t, large_t, out1, out2
gc.collect()
# 同步所有流，确保Event状态可被allocator检测
torch_npu.npu.synchronize()
# empty_cache 触发内存池归还逻辑，打印pool/Event日志
torch_npu.npu.empty_cache()
print("执行完成，查看终端memory日志")
```

### 执行代码

```Bash
# 1. 开启memory模块日志
export TORCH_NPU_LOGS=+memory
# 2. 设置日志输出级别 INFO
export TORCH_NPU_LOG_LEVEL=TRACE
# 再运行python脚本
python l6_3.py
```