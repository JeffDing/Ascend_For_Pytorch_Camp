import torch

def test_three_accelerator_api():
    print("========== torch.accelerator 三接口测试（NPU后端） ==========\n")

    # 接口1：判断加速器是否可用
    avail = torch.accelerator.is_available()
    print(f"1. torch.accelerator.is_available() = {avail}")

    if avail:
        # 接口2：获取NPU设备总数
        dev_total = torch.accelerator.device_count()
        print(f"2. torch.accelerator.device_count() = {dev_total}")

        # 切换设备（消除弃用警告）
        torch.accelerator.set_device_index(0)

        # 接口3：获取当前设备编号（标准新接口，无警告）
        cur_dev_idx = torch.accelerator.current_device_index()
        print(f"3. torch.accelerator.current_device_index() = {cur_dev_idx}")

        # 张量校验改用原生npu写法，避开不存在的accelerator.device
        t = torch.rand(2, 2, device=f"npu:{cur_dev_idx}")
        print(f"\n张量校验：tensor.device = {t.device}")
    else:
        print("当前无可用NPU加速器，请检查torch_npu、驱动、固件环境")

if __name__ == "__main__":
    test_three_accelerator_api()