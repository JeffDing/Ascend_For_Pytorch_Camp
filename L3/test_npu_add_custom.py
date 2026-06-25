import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

class TestCustomAdd(TestCase):

    def test_add_custom(self):
        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float16)
        y = torch.rand(length, device='cpu', dtype=torch.float16)

        output = torch_npu.npu_add_custom(x.npu(), y.npu()).cpu()
        self.assertRtolEqual(output, x + y)

    def test_add_custom_backward(self):
        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float16, requires_grad=True)
        y = torch.rand(length, device='cpu', dtype=torch.float16, requires_grad=True)

        output = torch_npu.npu_add_custom(x.npu(), y.npu()).cpu()
        grad_output = torch.rand(length, device='cpu', dtype=torch.float16)

        x_grad, y_grad = torch_npu.npu_add_custom_backward(grad_output.npu())
        self.assertRtolEqual(x_grad.cpu(), grad_output)
        self.assertRtolEqual(y_grad.cpu(), grad_output)

if __name__ == "__main__":
   run_tests()
