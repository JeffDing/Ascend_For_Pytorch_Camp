"""步骤 1：eager 基线。

先直接运行并记录耗时；然后复制本文件，分别改造成两个图模式版本。
具体要求和参考答案见 README 与 reference/ 目录。
"""

import time

import torch


DEVICE = 0
NUMEL = 65_536
OPS = 50
WARMUP = 5
ITERS = 50
SEED = 2026


def workload(x, steps=OPS):
    """由 add、mul、relu 组成的一串小 pointwise 算子。"""
    y = x
    for index in range(steps):
        if index % 3 == 0:
            y = y + 0.01
        elif index % 3 == 1:
            y = y * 0.999
        else:
            y = torch.relu(y)
    return y


def average_ms(run_once, prepare, synchronize, iterations):
    """同步后测量多轮执行的平均耗时。"""
    synchronize()
    start = time.perf_counter()
    for index in range(iterations):
        prepare(index)
        run_once()
    synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def main():
    import torch_npu  # noqa: F401  注册 torch.npu

    torch.npu.set_device(DEVICE)
    torch.manual_seed(SEED)
    torch.npu.manual_seed_all(SEED)

    device = f"npu:{DEVICE}"
    sources = [torch.randn(NUMEL, device=device) for _ in range(4)]
    input_buffer = torch.empty_like(sources[0])

    def prepare(index):
        # 三种实现都包含这次 copy_，保证比较口径一致。
        input_buffer.copy_(sources[index % len(sources)])

    def run_once():
        return workload(input_buffer)

    with torch.inference_mode():
        for index in range(WARMUP):
            prepare(index)
            output = run_once()
        torch.npu.synchronize()

        latency_ms = average_ms(
            run_once, prepare, torch.npu.synchronize, ITERS
        )
        prepare(0)
        output = run_once()
        torch.npu.synchronize()

    print(f"mode=eager  latency={latency_ms:.4f} ms/iter")
    print(f"output_mean={output.float().mean().item():.6f}")


if __name__ == "__main__":
    main()
