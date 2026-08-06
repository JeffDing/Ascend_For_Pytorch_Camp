"""步骤 2：torch.compile reduce-overhead 模式（torchair/GE 后端）。

- 用 torch.compile(..., backend="npu", mode="reduce-overhead") 包装 workload，
  走 torchair GE 图引擎，不依赖 triton-ascend。
- run_once() 调用编译后的函数。
- 编译、预热和图录制阶段不计入稳态耗时。
- 计时前先把输出与 eager 结果做 torch.testing.assert_close。
- 预热后和校验后各打印一行确认信息，确认是否真正进图。
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

    # 用 backend="npu"（torchair/GE 图引擎）+ reduce-overhead 模式编译，
    # 不依赖 triton-ascend，走 CANN GE 的图录制与重放。
    compiled_workload = torch.compile(
        workload, backend="npu", mode="reduce-overhead"
    )

    def prepare(index):
        # 三种实现都包含这次 copy_，保证比较口径一致。
        input_buffer.copy_(sources[index % len(sources)])

    def run_once():
        return compiled_workload(input_buffer)

    with torch.inference_mode():
        # (1) eager 基线，用于正确性比对。
        prepare(0)
        eager_output = workload(input_buffer)
        torch.npu.synchronize()

        # (2) 编译 + 预热 + 图录制：第一次调用触发编译，
        #     后续 WARMUP 轮让 reduce-overhead 的图录制完成。
        #     这部分耗时不算稳态，不进入计时。
        for index in range(WARMUP):
            prepare(index)
            run_once()
        torch.npu.synchronize()
        print("[info] graph compiled and warmed up")

        # (3) 正确性校验：在正式计时前与 eager 结果做 assert_close。
        prepare(0)
        compiled_output = run_once()
        torch.npu.synchronize()
        torch.testing.assert_close(compiled_output, eager_output)
        print("[info] assert_close passed")

        # (4) 稳态计时：只统计这一段的平均耗时。
        latency_ms = average_ms(
            run_once, prepare, torch.npu.synchronize, ITERS
        )

        # (5) 取一份最终输出用于打印。
        prepare(0)
        output = run_once()
        torch.npu.synchronize()

    print(f"mode=torch_compile(backend=npu,reduce-overhead)  latency={latency_ms:.4f} ms/iter")
    print(f"output_mean={output.float().mean().item():.6f}")


if __name__ == "__main__":
    main()
