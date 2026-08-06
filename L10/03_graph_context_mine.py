"""步骤 3：手写 NPU graph() 版本。

- 创建不会换地址的 static_input。
- 在非默认流上预热。
- 创建 torch.npu.NPUGraph()，用 torch.npu.graph(...) 捕获一次。
- 每轮用 static_input.copy_(new_input) 更新内容。
- graph.replay() 后读取捕获时的 static_output。
- 先与 eager 输出做 torch.testing.assert_close，再计时。
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

    # (1) 创建不会换地址的 static_input，图捕获和重放始终使用这块内存。
    static_input = torch.empty_like(sources[0])

    with torch.inference_mode():
        # (2) 在非默认流上预热，让算子初始化完成。
        side_stream = torch.npu.Stream()
        side_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(side_stream):
            for index in range(WARMUP):
                static_input.copy_(sources[index % len(sources)])
                workload(static_input)
            torch.npu.current_stream().wait_stream(side_stream)
        torch.npu.synchronize()

        # (3) 创建 NPUGraph，用 torch.npu.graph(...) 捕获一次。
        npu_graph = torch.npu.NPUGraph()
        with torch.npu.graph(npu_graph):
            static_output = workload(static_input)

        # (4) 每轮用 static_input.copy_(new_input) 更新内容。
        def prepare(index):
            static_input.copy_(sources[index % len(sources)])

        # (5) graph.replay() 后读取捕获时的 static_output。
        def run_once():
            npu_graph.replay()
            return static_output

        # eager 基线，用于正确性比对。
        prepare(0)
        eager_output = workload(static_input)
        torch.npu.synchronize()

        # (6) 先与 eager 输出做 assert_close，再计时。
        prepare(0)
        run_once()
        torch.npu.synchronize()
        torch.testing.assert_close(static_output, eager_output)

        # 稳态计时。
        latency_ms = average_ms(
            run_once, prepare, torch.npu.synchronize, ITERS
        )

        # 取一份最终输出用于打印。
        prepare(0)
        run_once()
        torch.npu.synchronize()
        output = static_output

    print(f"mode=graph_context  latency={latency_ms:.4f} ms/iter")
    print(f"output_mean={output.float().mean().item():.6f}")


if __name__ == "__main__":
    main()
