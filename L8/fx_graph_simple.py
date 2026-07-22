import torch
import torch._dynamo
from torch._dynamo.backends.common import aot_autograd

def loss_fn(x, target):
    pred = torch.sigmoid(x)
    mask = pred > 0.5
    masked = pred * mask
    loss = torch.nn.functional.binary_cross_entropy(pred, target)
    return loss, masked

x = torch.randn(8, 16, requires_grad=True)
target = torch.empty(8, 16).random_(0, 2)

graphs = []
def fw_compiler(gm, inps):
    graphs.append(gm)
    gm.graph.print_tabular()
    return gm

backend = aot_autograd(fw_compiler=fw_compiler)
compiled_fn = torch.compile(loss_fn, backend=backend)
loss, masked = compiled_fn(x, target)
loss.backward()