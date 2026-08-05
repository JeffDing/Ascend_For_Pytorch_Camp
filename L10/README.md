# npugraph 课后练习

这份练习不要求先读懂一个完整框架。你只需要从 eager 基线出发，自己改出两个版本：

1. `torch.compile(mode="reduce-overhead")`
2. 手写 `torch.npu.graph()`

完成后再打开 `reference/` 对照参考实现。

## 目录

```text
npugraph_homework/
├── README.md
├── 01_eager.py                 # 学员起始文件
└── reference/
    ├── 02_torch_compile.py     # 参考答案 1
    └── 03_graph_context.py     # 参考答案 2
```

## 第一步：建立 eager 基线

```bash
python 01_eager.py
```

记录输出中的平均耗时和 `output_mean`。三个版本使用相同的输入、workload、预热次数和计时方法。

## 第二步：自己改出 torch.compile 版本

先复制 eager 文件，不要直接看参考答案：

```bash
cp 01_eager.py 02_torch_compile_mine.py
```

改造目标：

- 用 `torch.compile(..., mode="reduce-overhead")` 包装 `workload`。
- `run_once()` 调用编译后的函数。
- 编译、预热和录制阶段不计入稳态耗时。
- 计时前先把输出与 eager 结果做 `torch.testing.assert_close`。
- 保存当前版本的 record/replay 或 skip 日志，确认是否真正进图。

完成后再对照：

```bash
python reference/02_torch_compile.py
```

## 第三步：自己改出手写 graph() 版本

再次从 eager 文件开始：

```bash
cp 01_eager.py 03_graph_context_mine.py
```

改造目标：

- 创建不会换地址的 `static_input`。
- 在非默认流上预热。
- 创建 `torch.npu.NPUGraph()`，用 `torch.npu.graph(...)` 捕获一次。
- 每轮用 `static_input.copy_(new_input)` 更新内容。
- `graph.replay()` 后读取捕获时的 `static_output`。
- 先与 eager 输出做 `torch.testing.assert_close`，再计时。

完成后再对照：

```bash
python reference/03_graph_context.py
```

## 第四步：整理结果

| 模式 | 平均耗时（ms/iter） | 相对 eager 加速倍数 | 是否确认 replay |
|---|---:|---:|---|
| eager |  | 1.000× | 不适用 |
| torch.compile |  |  |  |
| graph() |  |  |  |

加速倍数使用 `eager 耗时 / 当前模式耗时`。固定运行几次不是进入稳态 replay 的充分证据，请同时提交日志。

## 思考题

1. eager 模式的 Host 开销来自哪里？两种图模式如何减少这部分开销？
2. `torch.compile` 与手写 `graph()` 的首轮成本和稳态耗时为什么可能不同？
3. 把 `NUMEL` 增大后，加速倍数如何变化？这说明 workload 更偏 Host Bound 还是 Device Bound？
4. 如果 compile 版本只有 skip、或 record 次数持续增加，你会沿哪条排障路径继续检查？

## 提交内容

- 上面的性能表。
- 一段能说明 skip、record 或 replay 状态的关键日志。
- 四道思考题的简短回答。
- 你自己完成的 `02_torch_compile_mine.py` 和 `03_graph_context_mine.py`。

> `WARMUP = 5` 只是本练习的最低预热次数，不是跨模型、跨版本的固定协议。培训现场使用的 PyTorch、torch_npu、CANN 与 NPU 型号应一并记录。
