"""参考答案 2：从 eager 改成手写 torch.npu.graph()。"""

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

    with torch.inference_mode():
        reference = workload(sources[0])
        torch.npu.synchronize()

        # 新增 1：这块地址会被图捕获，之后只更新内容，不替换 Tensor。
        static_input = torch.empty_like(sources[0])
        static_input.copy_(sources[0])

        # 新增 2：在非默认流上预热，使初始化和分配趋于稳定。
        capture_stream = torch.npu.Stream()
        capture_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(capture_stream):
            for _ in range(WARMUP):
                workload(static_input)
        torch.npu.current_stream().wait_stream(capture_stream)
        torch.npu.synchronize()

        # 新增 3：只捕获一次。
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=capture_stream):
            static_output = workload(static_input)
        torch.npu.synchronize()

        def prepare(index):
            # 新数据写进捕获时的固定地址。
            static_input.copy_(sources[index % len(sources)])

        def run_once():
            graph.replay()
            return static_output

        prepare(0)
        output = run_once()
        torch.npu.synchronize()
        torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-3)

        latency_ms = average_ms(
            run_once, prepare, torch.npu.synchronize, ITERS
        )

    print(f"mode=graph  latency={latency_ms:.4f} ms/iter")
    print(f"output_mean={output.float().mean().item():.6f}")


if __name__ == "__main__":
    main()
