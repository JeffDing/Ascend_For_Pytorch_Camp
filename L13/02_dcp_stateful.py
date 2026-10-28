#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCP 课后作业 2 —— Stateful 协议: 一个类两个方法, 打包保存与恢复完整训练状态。

实现 StatefulTrainer 的 state_dict / load_state_dict, DCP 就会自动识别并调用:
模型权重、优化器动量、step 计数一起保存, 一起恢复 —— 这就是完整续训状态,
而不只是"权重级恢复"。

运行方式 (CPU, 默认 Gloo):
    torchrun --standalone --nproc-per-node=2 02_dcp_stateful.py

使用 GPU 或 NPU 时 (不超过 2 卡), 在命令末尾添加 --device cuda 或 --device npu。

模板提供进程组初始化、固定数据、假训练、warm-up、打印与验收断言;
只需补齐 TODO 函数, 删除对应的 raise NotImplementedError(...), 无需修改其他代码。
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "npu"])
    parser.add_argument("--ckpt-dir", default="./ckpt_stateful")
    return parser.parse_args()


def init_env(device_type: str) -> None:
    """初始化进程组。NPU 需先导入 torch_npu 并 transfer_to_npu (否则 device_count=0)。"""
    if device_type == "npu":
        import torch_npu  # noqa: F401
        torch_npu.transfer_to_npu()
        backend = "hccl"
    elif device_type == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)


def get_dev(device_type: str) -> torch.device:
    """每个进程使用 LOCAL_RANK 对应的那张卡; CPU 无所谓。"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "cuda":
        return torch.device(f"cuda:{local_rank}")
    if device_type == "npu":
        return torch.device(f"npu:{local_rank}")
    return torch.device("cpu")


def build_model(device: torch.device) -> nn.Module:
    """固定随机种子, 两次调用得到完全相同的初始模型。"""
    torch.manual_seed(1234)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    return model.to(device)


def fake_train(trainer: "StatefulTrainer", device: torch.device, steps: int = 3) -> None:
    """固定输入的假训练: 3 步之后 Adam 的 exp_avg / exp_avg_sq / step 皆有值。"""
    torch.manual_seed(2024)
    x = torch.randn(steps, 4, 16, device=device)
    y = torch.randn(steps, 4, 4, device=device)
    for i in range(steps):
        loss = F.mse_loss(trainer.model(x[i]), y[i])
        trainer.optim.zero_grad()
        loss.backward()
        trainer.optim.step()


def warm_up_optimizer(trainer: "StatefulTrainer") -> None:
    """给"还没走过 step"的新优化器补一次零梯度 step:
    权重不会变 (梯度全 0), 但 Adam 会提前创建 exp_avg / exp_avg_sq / step 这些张量。
    DCP 加载是原地填充, 目的地张量必须先存在 —— 为什么要这样, 见 README 简答题。"""
    for p in trainer.model.parameters():
        p.grad = torch.zeros_like(p)
    trainer.optim.step()
    trainer.optim.zero_grad(set_to_none=True)


def log(msg: str) -> None:
    print(f"[rank{dist.get_rank()}] {msg}", flush=True)


class StatefulTrainer:
    """TODO: 实现下面两个方法, 这个类就满足了 DCP 的 Stateful 协议
    (只要有 state_dict() 和 load_state_dict() 就行, 不要求继承任何基类)。"""

    def __init__(self, model: nn.Module, optim: torch.optim.Optimizer):
        self.model = model
        self.optim = optim

    def state_dict(self) -> dict:
        """TODO-1: 返回 {"model": ..., "optim": ...} 的嵌套 state_dict。"""
        return {"model": self.model.state_dict(), "optim": self.optim.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        """TODO-2: 从 state_dict 恢复 model 与 optim。"""
        self.model.load_state_dict(state_dict["model"])
        self.optim.load_state_dict(state_dict["optim"])


# ===================== 以上是 TODO 区 =====================


def save_trainer(trainer: StatefulTrainer, ckpt_dir: str) -> None:
    """TODO-3: 调用 dcp.save 把 trainer (Stateful 对象) 保存到 ckpt_dir。

    提示: state_dict 写成 {"trainer": trainer}。
    """
    dcp.save({"trainer": trainer}, checkpoint_id=ckpt_dir)


def load_trainer(trainer: StatefulTrainer, ckpt_dir: str) -> None:
    """TODO-4: 调用 dcp.load 恢复 trainer。"""
    dcp.load({"trainer": trainer}, checkpoint_id=ckpt_dir)


# ===================== 以下是验收流程, 无需修改 =====================


def snapshot(trainer: StatefulTrainer) -> dict:
    """给断言用的快照: 以参数名为键, 保存权重、Adam 动量、step、学习率的副本。"""
    weight, optim_state = {}, {}
    for name, p in trainer.model.named_parameters():
        weight[name] = p.detach().clone()
        s = trainer.optim.state[p]
        optim_state[name] = {k: v.clone() for k, v in s.items() if torch.is_tensor(v)}
    return {
        "weight": weight,
        "optim_state": optim_state,
        "lr": trainer.optim.param_groups[0]["lr"],
    }


def verify(before: dict, after_trainer: StatefulTrainer) -> None:
    for name, p in after_trainer.model.named_parameters():
        assert torch.allclose(p.detach(), before["weight"][name]), f"权重不一致: {name}"
        saved = before["optim_state"][name]
        cur = after_trainer.optim.state[p]
        assert set(saved) <= set(cur), f"恢复后优化器状态字段缺失: {name}"
        for k, v in saved.items():
            assert torch.allclose(cur[k], v), f"优化器状态 {name}.{k} 不一致"
    assert after_trainer.optim.param_groups[0]["lr"] == before["lr"], "lr 没有恢复"


def main() -> None:
    args = parse_args()
    device = get_dev(args.device)
    init_env(args.device)
    try:
        # ---- 1. 训练 3 步, 让 Adam 产生真实状态, 然后保存 ----
        model1 = build_model(device)
        trainer1 = StatefulTrainer(model1, torch.optim.Adam(model1.parameters(), lr=1e-2))
        fake_train(trainer1, device)
        assert trainer1.optim.state, "假训练后优化器没有状态, 请检查 fake_train 是否被改动"

        save_trainer(trainer1, args.ckpt_dir)              # <-- TODO-3
        before = snapshot(trainer1)
        log(f"保存完成: 训练状态已写入 {args.ckpt_dir} (lr={before['lr']})")

        # ---- 2. 新模型 + 新优化器, 先 warm-up 出目的地, 再原地恢复 ----
        model2 = build_model(device)
        trainer2 = StatefulTrainer(model2, torch.optim.Adam(model2.parameters(), lr=1e-2))
        warm_up_optimizer(trainer2)

        load_trainer(trainer2, args.ckpt_dir)               # <-- TODO-4
        verify(before, trainer2)
        log("恢复完成: 权重 / exp_avg / step / lr 全部一致")
        log("PASS: 作业 2 全部断言通过")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
