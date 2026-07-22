import os
os.environ["TORCH_LOGS"] = "dynamo,aot,graph_code"
os.environ["TORCH_DYNAMO_LOG_LEVEL"] = "INFO"
os.environ["TORCH_DYNAMO_PRINT_GRAPH_BREAKS"] = "1"

import torch
from torch.fx import GraphModule, symbolic_trace

def loss_fn(x, target):
    pred = torch.sigmoid(x)
    mask = pred > 0.5
    masked = pred * mask
    loss = torch.nn.functional.binary_cross_entropy(pred, target)
    return loss, masked

def print_fx_graph(gm: GraphModule, title: str, save_file: bool = False):
    """统一打印FX表格与源码，可选保存文件"""
    border = "=" * 75
    print(f"\n{border}")
    print(f"【{title}】")
    print(f"{border}")
    gm.graph.print_tabular()
    print("\n>>>> Graph Python Source Code:")
    print(gm.code)

    if save_file:
        filename = f"{title.replace(' ', '_').replace('#','')}.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(gm.code)
        print(f"\n>>>> Graph code saved to: {filename}")

if __name__ == "__main__":
    # ==============================================
    # 1. symbolic_trace: Eager模式原生前向FX（基线对比）
    # ==============================================
    print("\n" + "#"*80)
    print("【阶段1】symbolic_trace EAGER 原生正向FX（仅作基线对比，不经过Dynamo）")
    print("#"*80)
    x_trace = torch.randn(8, 16)
    t_trace = torch.randint(0, 2, (8, 16)).float()

    def wrap_func(x, target):
        return loss_fn(x, target)

    base_gm = symbolic_trace(wrap_func)
    print_fx_graph(base_gm, "基线_Eager_Original_Forward_FX", save_file=False)

    # ==============================================
    # 2. torch._dynamo.explain：Dynamo捕获Compile入口正向FX（核心）
    # 只追踪计算图，不会触发NPU后端编译，规避算子Converter缺失报错
    # ==============================================
    print("\n\n" + "#"*80)
    print("【阶段2】Dynamo 捕获 torch.compile 输入正向FX Graph")
    print("#"*80)
    x_cpu = torch.randn(8, 16, requires_grad=True)
    target_cpu = torch.randint(0, 2, (8, 16)).float()

    explain_out = torch._dynamo.explain(loss_fn)(x_cpu, target_cpu)
    captured_graphs = explain_out.graphs

    for idx, gm in enumerate(captured_graphs):
        if isinstance(gm, GraphModule):
            print_fx_graph(gm, f"Dynamo_Captured_Forward_FX_{idx}", save_file=False)

    # ==============================================
    # 3. 关于反向FX重要说明（重点）
    # ==============================================
    print("\n\n" + "!"*80)
    print("【重要约束说明 - 反向FX Graph】")
    print("!"*80)
    print("1. 当前 PyTorch 2.7.1 + torch-npu 2.7.1 无对外Python API，")
    print("   无法获取AOTAutograd拆分后的【反向FX GraphModule对象】；")
    print("2. 反向计算图在C++内部生成，仅能通过日志查看算子序列文本；")
    print("3. 控制台日志检索关键词：")
    print("      > 正向图日志：TRACE dynamo: GraphModule captured")
    print("      > AOT反向算子日志：backward graph")
    print("4. 不存在可行方式构造可编程操作的反向FX Graph！")
    print("!"*80)

    # ==============================================
    # 4. NPU Eager 执行（不使用torch.compile，稳定跑通训练流程）
    # ==============================================
    print("\n\n" + "#"*80)
    print("【阶段3】NPU Eager模式 前向+反向正常执行（无compile）")
    print("#"*80)
    device = torch.device("npu")
    x_npu = torch.randn(8, 16, requires_grad=True, device=device)
    target_npu = torch.empty(8, 16, device=device).random_(0, 2)
    opt = torch.optim.SGD([x_npu], lr=0.01)

    loss_npu, masked_npu = loss_fn(x_npu, target_npu)
    loss_npu.backward()
    opt.step()
    print(f"NPU Eager forward loss value = {loss_npu.item():.6f}")