import torch

def test_to_npu_raii():
    print("==== Start tensor.to('npu') RAII test ====")
    cpu_t = torch.randn(2, 4)
    # 构造NPUGuard：调用exchangeDevice切到npu
    npu_t = cpu_t.to("npu:0")
    print(f"tensor device: {npu_t.device}")
    # 函数作用域结束，NPUGuard析构，再次exchangeDevice切回原始设备
    print("==== Scope exit, NPUGuard destruct trigger device restore ====")

if __name__ == "__main__":
    # 开启NPU INFO日志
    import os
    os.environ["TORCH_NPU_LOG_LEVEL"] = "INFO"
    test_to_npu_raii()