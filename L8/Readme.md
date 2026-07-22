# 任务
1. 根据调试命令，获取正反向FX图
2. mask = pred > 0.5 这条比较操作会被捕获到 FX 图中吗？为什么？

```python
import torch
def loss_fn(x, target):
pred = torch.sigmoid(x)
    mask = pred > 0.5
    masked = pred * mask
    loss = torch.nn.functional.binary_cross_entropy(pred, target)
    return loss, masked
    x = torch.randn(8, 16, requires_grad=True)
target = torch.empty(8, 16).random_(0, 2)
opt = torch.optim.SGD([x], lr=0.01)

compiled_fn = torch.compile(loss_fn, backend="inductor")
loss, masked = compiled_fn(x, target)
loss.backward()
opt.step()
```