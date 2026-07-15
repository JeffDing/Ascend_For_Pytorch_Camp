# Ascend for Pytroch第七课

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

## 作业一：算子下发调用链路分析

示例仅供参考，请基于其他api进行链路分析

```Python
import torch
import torch_npu
def test_npu_torch_add():
    # 1. 环境校验
    print("=== NPU环境校验 ===")
    print(f"torch_npu 版本: {torch_npu.version}")
    print(f"NPU可用: {torch.npu.is_available()}")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    print(f"使用设备: {device}\n")
    # 2. 张量+标量 NPU运算
    print("=== 1. NPU张量 + 标量 ===")
    a = torch.tensor([1, 2, 3], device=device, dtype=torch.float32)
    with torch.autograd.profiler.profile(record_shapes=True) as prof:
        res1 = torch.add(a, 5.0)
    print(prof.key_averages(group_by_input_shape=True))
    print(f"torch.add(a, 5) = {res1}")
    print(f"设备校验: res1.device = {res1.device}\n")
```

选用 `torch.mean`：

```Python
import torch
import torch_npu

def test_npu_torch_mean():
    # 1. 环境校验
    print("=== NPU环境校验 ===")
    print(f"torch_npu 版本: {torch_npu.version}")
    print(f"NPU可用: {torch.npu.is_available()}")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    print(f"使用设备: {device}\n")

    # 2. NPU张量均值运算，指定维度求平均
    print("=== 1. NPU张量 torch.mean 维度规约运算 ===")
    a = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], device=device, dtype=torch.float32)
    
    with torch.autograd.profiler.profile(record_shapes=True) as prof:
        # dim=1：按行求均值；keepdim=False 不保留规约维度
        res1 = torch.mean(a, dim=1)
    
    print(prof.key_averages(group_by_input_shape=True))
    print(f"torch.mean(a, dim=1) = {res1}")
    print(f"设备校验: res1.device = {res1.device}\n")

if __name__ == "__main__":
    test_npu_torch_mean()
```

## 作业二：使能TASK\_QUEUE\_ENABLE，查看耗时

示例仅供参考，请构造其他场景进行taskqueue使能

```Python
import torch
import torch_npu
import time
def test_npu_torch_add():
    # 1. 环境校验
    print("=== NPU环境校验 ===")
    print(f"torch_npu 版本: {torch_npu.**version**}")
    print(f"NPU可用: {torch.npu.is_available()}")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    print(f"使用设备: {device}\n")
    # 2. 超大尺寸张量，增加单次运算耗时
    # 2048*2048 = 4,194,304个浮点元素，计算量充足
    tensor_shape = (2048, 2048)
    a = torch.randn(tensor_shape, device=device, dtype=torch.float32)
    
    # 预热：消除算子编译、内存初始化开销
    warmup_iters = 200
    for _ in range(warmup_iters):
        _ = torch.add(a, 5.0)
    torch.npu.synchronize()  # 等待预热全部完成
    
    # 正式循环计时
    loop_cnt = 1000000
    start_time = time.time()
    for _ in range(loop_cnt):
        res1 = torch.add(a, 5.0)
    torch.npu.synchronize()  # NPU异步，必须同步后再结束计时
    end_time = time.time()
    
    total_cost = end_time - start_time
    avg_sec = total_cost / loop_cnt
    avg_ms = avg_sec * 1000
    
    print(f"张量Shape: {tensor_shape}")
    print(f"总循环次数: {loop_cnt}")
    print(f"总耗时: {total_cost:.4f} s")
    print(f"单次平均耗时: {avg_sec:.8f} s / {avg_ms:.4f} ms")
    print(f"结果设备校验: res1.device = {res1.device}")
if **name** == "**main**":
    test_npu_torch_add()
```

操作：

选用 `torch.relu` 激活算子

```Python
import os
import torch
import torch_npu
import time

def test_npu_torch_relu():
    # =========【关键】控制TaskQueue开关 =========
    # 运行前设置环境变量，也可脚本外部export
    # os.environ["TASK_QUEUE_ENABLE"] = "1"  # 开启TaskQueue
    # os.environ["TASK_QUEUE_ENABLE"] = "0"  # 关闭TaskQueue
    print(f"TASK_QUEUE_ENABLE = {os.environ.get('TASK_QUEUE_ENABLE', 'not set')}\n")

    # 1. 环境校验
    print("=== NPU环境校验 ===")
    print(f"torch_npu 版本: {torch_npu.version}")
    print(f"NPU可用: {torch.npu.is_available()}")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    print(f"使用设备: {device}\n")

    # 2. 构造测试张量
    tensor_shape = (2048, 2048)
    a = torch.randn(tensor_shape, device=device, dtype=torch.float32)

    # 预热：消除算子编译、内存初始化开销
    warmup_iters = 200
    for _ in range(warmup_iters):
        _ = torch.relu(a)
    torch.npu.synchronize()
    print("预热完成\n")

    # 正式循环压测
    loop_cnt = 200000
    start_time = time.time()
    for _ in range(loop_cnt):
        res = torch.relu(a)
    torch.npu.synchronize()  # 异步任务必须同步，保证全部执行完成
    end_time = time.time()

    total_cost = end_time - start_time
    avg_sec = total_cost / loop_cnt
    avg_ms = avg_sec * 1000

    print(f"张量Shape: {tensor_shape}")
    print(f"循环次数: {loop_cnt}")
    print(f"总耗时: {total_cost:.4f} s")
    print(f"单次算子平均耗时: {avg_sec:.8f} s / {avg_ms:.6f} ms")
    print(f"输出张量设备: {res.device}")

if __name__ == "__main__":
    test_npu_torch_relu()
```
