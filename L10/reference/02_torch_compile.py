"""参考答案 1：从 eager 改成 torch.compile(reduce-overhead)。"""

import time

import torch


DEVICE = 0
NUMEL = 65_536
OPS = 50
WARMUP = 5
ITERS = 50
SEED = 2026


def workload(x, steps=OPS):
    """与 01_eager.py 完全相同的 workload。"""
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
    """与 01_eager.py 完全相同的计时方法。"""
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
        input_buffer.copy_(sources[index % len(sources)])

    # 新增 1：让编译器接管图捕获与重放。
    compiled_workload = torch.compile(workload, mode="reduce-overhead")

    def run_once():
        return compiled_workload(input_buffer)

    with torch.inference_mode():
        prepare(0)
        reference = workload(input_buffer)
        torch.npu.synchronize()

        # 新增 2：编译、预热和录制阶段不计入稳态耗时。
        for index in range(WARMUP):
            prepare(index)
            output = run_once()
        torch.npu.synchronize()

        prepare(0)
        output = run_once()
        torch.npu.synchronize()
        torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-3)

        latency_ms = average_ms(
            run_once, prepare, torch.npu.synchronize, ITERS
        )

    print(f"mode=compile  latency={latency_ms:.4f} ms/iter")
    print(f"output_mean={output.float().mean().item():.6f}")
    print("请结合当前版本日志确认已经进入稳定 replay。")


if __name__ == "__main__":
    main()
