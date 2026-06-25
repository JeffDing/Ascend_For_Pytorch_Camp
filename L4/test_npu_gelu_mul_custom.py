#!/usr/bin/env python3
"""
npu_gelu_mul_custom 算子验证脚本

功能：验证自定义算子 npu_gelu_mul_custom 的正确性
  - 与前向 GELU + Mul 组合计算结果进行精度比对
  - 覆盖 approximate="none"（erf 高精度模式）
  - 覆盖 approximate="tanh"（tanh 近似模式）
  - 覆盖 float16 / float32 数据类型
  - 检查输出 shape 正确性
  - 支持 FP32 相对精度 1e-3、FP16 相对精度 1e-2 的容差

使用方式：
  python test_npu_gelu_mul_custom.py

依赖环境：
  - PyTorch (>=2.1.0)
  - torch_npu (已安装并支持 NPU 设备)
"""

import sys
import numpy as np
import torch
import torch_npu


def get_golden_output(input_tensor, approximate="none"):
    """
    计算 CPU 上 GELU + Mul 组合的参考输出。

    GELU 公式：
      approximate="none":  GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
      approximate="tanh":  GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    计算流程：
      1. 沿最后一维将 input 拆分为 x1, x2（各占一半）
      2. 对 x1 应用 GELU 激活函数
      3. out = GELU(x1) * x2
    """
    last_dim = input_tensor.shape[-1]
    if last_dim % 2 == 1:
        raise ValueError(f"Input last dimension must be even, got {last_dim}")

    d = last_dim // 2
    x1 = input_tensor[..., :d]
    x2 = input_tensor[..., d:]
    gelu = torch.nn.GELU(approximate=approximate)
    x1 = gelu(x1)
    output = x1 * x2
    return output


def test_npu_gelu_mul_custom_basic():
    """
    测试 1：基础功能验证
    - 调用 npu_gelu_mul_custom
    - 检查输出不为 None
    - 检查输出 shape 正确性（最后一维应为输入一半）
    """
    print("=" * 60)
    print("测试 1：基础功能验证")
    print("=" * 60)

    shape = [100, 400]
    input_tensor = torch.rand(shape, dtype=torch.float16).npu()

    try:
        output = torch_npu.npu_gelu_mul_custom(input_tensor, approximate="tanh")
        assert output is not None, "输出为 None"
        assert output.shape[-1] == input_tensor.shape[-1] // 2, \
            f"输出最后一维应为 {input_tensor.shape[-1] // 2}，实际为 {output.shape[-1]}"
        assert output.dtype == input_tensor.dtype, \
            f"输出 dtype 应与输入一致，{output.dtype} != {input_tensor.dtype}"
        assert output.device.type == "npu", f"输出应在 NPU 设备上，实际为 {output.device.type}"
        print(f"  [PASS] 基础功能验证通过")
        print(f"     输入 shape: {tuple(input_tensor.shape)}, dtype: {input_tensor.dtype}")
        print(f"     输出 shape: {tuple(output.shape)}, dtype: {output.dtype}")
    except Exception as e:
        print(f"  [FAIL] 基础功能验证失败: {e}")
        return False
    return True


def test_npu_gelu_mul_custom_approximate_none():
    """
    测试 2：approximate="none" 模式精度验证
    - 使用 erf 高精度 GELU 公式
    - 与 CPU 参考结果逐元素比对
    - float32 精度：rtol=1e-3, atol=1e-3
    - float16 精度：rtol=1e-2, atol=1e-2
    """
    print("\n" + "=" * 60)
    print("测试 2：approximate='none' 模式精度验证")
    print("=" * 60)

    shapes = [
        [100, 400],
        [32, 64, 128],
        [16, 32, 64, 256],
    ]
    dtypes = [torch.float32, torch.float16]
    all_pass = True

    for shape in shapes:
        for dtype in dtypes:
            rtol = 1e-3 if dtype == torch.float32 else 1e-2
            atol = 1e-3 if dtype == torch.float32 else 1e-2

            input_tensor = torch.rand(shape, dtype=dtype).npu()
            try:
                output = torch_npu.npu_gelu_mul_custom(input_tensor, approximate="none")
                golden = get_golden_output(input_tensor.cpu(), approximate="none")

                match = torch.allclose(output.cpu(), golden, rtol=rtol, atol=atol)
                if match:
                    print(f"  [PASS] shape={shape}, dtype={dtype} — 精度验证通过")
                else:
                    max_diff = (output.cpu() - golden).abs().max().item()
                    print(f"  [FAIL] shape={shape}, dtype={dtype} — 精度验证失败 (max_diff={max_diff:.6f})")
                    all_pass = False
            except Exception as e:
                print(f"  [FAIL] shape={shape}, dtype={dtype} — 调用异常: {e}")
                all_pass = False

    return all_pass


def test_npu_gelu_mul_custom_approximate_tanh():
    """
    测试 3：approximate="tanh" 模式精度验证
    - 使用 tanh 近似 GELU 公式
    - 与 CPU 参考结果逐元素比对
    """
    print("\n" + "=" * 60)
    print("测试 3：approximate='tanh' 模式精度验证")
    print("=" * 60)

    shapes = [
        [100, 400],
        [32, 64, 128],
        [16, 32, 64, 256],
    ]
    dtypes = [torch.float32, torch.float16]
    all_pass = True

    for shape in shapes:
        for dtype in dtypes:
            rtol = 1e-3 if dtype == torch.float32 else 1e-2
            atol = 1e-3 if dtype == torch.float32 else 1e-2

            input_tensor = torch.rand(shape, dtype=dtype).npu()
            try:
                output = torch_npu.npu_gelu_mul_custom(input_tensor, approximate="tanh")
                golden = get_golden_output(input_tensor.cpu(), approximate="tanh")

                match = torch.allclose(output.cpu(), golden, rtol=rtol, atol=atol)
                if match:
                    print(f"  [PASS] shape={shape}, dtype={dtype} — 精度验证通过")
                else:
                    max_diff = (output.cpu() - golden).abs().max().item()
                    print(f"  [FAIL] shape={shape}, dtype={dtype} — 精度验证失败 (max_diff={max_diff:.6f})")
                    all_pass = False
            except Exception as e:
                print(f"  [FAIL] shape={shape}, dtype={dtype} — 调用异常: {e}")
                all_pass = False

    return all_pass


def test_npu_gelu_mul_custom_non_contiguous():
    """
    测试 4：非连续张量支持验证
    - 使用 transpose 生成非连续张量
    - 验证结果正确性
    """
    print("\n" + "=" * 60)
    print("测试 4：非连续张量支持验证")
    print("=" * 60)

    shape = [64, 128, 256]
    dtype = torch.float16
    input_tensor = torch.rand(shape, dtype=dtype).npu()

    # 通过 transpose 创建非连续张量
    non_contiguous = input_tensor.transpose(1, 2).contiguous().transpose(1, 2)
    assert not non_contiguous.is_contiguous(), "预期为非连续张量"

    try:
        output = torch_npu.npu_gelu_mul_custom(non_contiguous, approximate="tanh")
        golden = get_golden_output(non_contiguous.cpu(), approximate="tanh")
        match = torch.allclose(output.cpu(), golden, rtol=1e-2, atol=1e-2)
        if match:
            print(f"  [PASS] 非连续张量验证通过")
            print(f"     输入 shape: {tuple(non_contiguous.shape)}, is_contiguous: {non_contiguous.is_contiguous()}")
            print(f"     输出 shape: {tuple(output.shape)}")
        else:
            max_diff = (output.cpu() - golden).abs().max().item()
            print(f"  [FAIL] 非连续张量验证失败 (max_diff={max_diff:.6f})")
            return False
    except Exception as e:
        print(f"  [FAIL] 非连续张量调用异常: {e}")
        return False

    return True


def test_npu_gelu_mul_custom_negative_values():
    """
    测试 5：负值输入验证
    - 测试负值输入下的 GELU 行为
    - GELU 在负值区域应有平滑的抑制效果
    """
    print("\n" + "=" * 60)
    print("测试 5：负值输入验证")
    print("=" * 60)

    shape = [100, 400]
    dtype = torch.float32
    # 生成包含负值的输入
    input_tensor = torch.randn(shape, dtype=dtype).npu()

    try:
        output = torch_npu.npu_gelu_mul_custom(input_tensor, approximate="tanh")
        golden = get_golden_output(input_tensor.cpu(), approximate="tanh")
        match = torch.allclose(output.cpu(), golden, rtol=1e-3, atol=1e-3)
        if match:
            print(f"  [PASS] 负值输入验证通过")
            print(f"     输入范围: [{input_tensor.cpu().min().item():.4f}, {input_tensor.cpu().max().item():.4f}]")
        else:
            max_diff = (output.cpu() - golden).abs().max().item()
            print(f"  [FAIL] 负值输入验证失败 (max_diff={max_diff:.6f})")
            return False
    except Exception as e:
        print(f"  [FAIL] 负值输入调用异常: {e}")
        return False

    return True


def test_npu_gelu_mul_custom_output_size():
    """
    测试 6：输出 shape 一致性验证
    - 验证输出最后一维 = 输入最后一维 / 2
    - 验证其他维度保持不变
    """
    print("\n" + "=" * 60)
    print("测试 6：输出 shape 一致性验证")
    print("=" * 60)

    test_shapes = [
        [10, 20],
        [32, 64],
        [16, 32, 128],
        [8, 16, 32, 256],
        [4, 8, 16, 32, 512],
    ]
    all_pass = True

    for shape in test_shapes:
        input_tensor = torch.rand(shape, dtype=torch.float32).npu()
        try:
            output = torch_npu.npu_gelu_mul_custom(input_tensor, approximate="none")
            expected_shape = list(shape)
            expected_shape[-1] //= 2
            actual_shape = list(output.shape)
            if actual_shape == expected_shape:
                print(f"  [PASS] shape {shape} → {actual_shape}")
            else:
                print(f"  [FAIL] shape {shape} → {actual_shape} (期望 {expected_shape})")
                all_pass = False
        except Exception as e:
            print(f"  [FAIL] shape {shape} — 异常: {e}")
            all_pass = False

    return all_pass


def main():
    """
    主函数：运行所有测试用例
    """
    print("\n" + "*" * 60)
    print("  npu_gelu_mul_custom 算子验证套件")
    print("*" * 60 + "\n")

    # 检查 NPU 是否可用
    if not torch.npu.is_available():
        print("[FAIL] NPU 不可用，请检查环境配置")
        sys.exit(1)
    print(f"  NPU 可用: {torch.npu.get_device_name(0)}")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  torch_npu 已加载: {hasattr(torch, 'npu') and torch.npu.is_available()}")

    # 检查算子是否已注册
    try:
        _ = torch.ops.npu.npu_gelu_mul_custom
        print(f"  [PASS] 算子 npu_gelu_mul_custom 已注册")
    except AttributeError:
        print(f"  [WARN]  算子 npu_gelu_mul_custom 未在 torch.ops.npu 中注册")
        print(f"     验证 torch_npu 是否已正确安装包含该算子的版本")

    print()

    # 运行测试
    results = {}
    results["test_basic"] = test_npu_gelu_mul_custom_basic()
    results["test_approximate_none"] = test_npu_gelu_mul_custom_approximate_none()
    results["test_approximate_tanh"] = test_npu_gelu_mul_custom_approximate_tanh()
    results["test_non_contiguous"] = test_npu_gelu_mul_custom_non_contiguous()
    results["test_negative_values"] = test_npu_gelu_mul_custom_negative_values()
    results["test_output_size"] = test_npu_gelu_mul_custom_output_size()

    # 汇总
    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    all_pass = True
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {test_name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  所有测试通过！")
    else:
        print("  [WARN]  部分测试未通过，请检查上述日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
