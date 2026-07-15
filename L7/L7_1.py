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