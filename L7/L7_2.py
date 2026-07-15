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