#include "op_plugin/OpApiInterface.h" 
#include "op_plugin/utils/op_api_common.h" 

namespace op_api { 
using npu_preparation = at_npu::native::OpPreparation;

// 正向接口
at::Tensor npu_add_custom(const at::Tensor& x, const at::Tensor& y, const at::Scalar &alpha)
{ 
    // 构造输出tensor 
    at::Tensor result = npu_preparation::apply_tensor_without_format(x);
    // 计算输出结果
    // 调用EXEC_NPU_CMD接口，完成输出结果的计算
    // 第一个入参格式为aclnn+Optype，之后的参数分别为输入输出
    EXEC_NPU_CMD(aclnnAdd, x, y, alpha, result); 
    return result; 
}

// 反向接口
std::tuple<at::Tensor, at::Tensor> npu_add_custom_backward(const at::Tensor& grad)
{
    // 构造输出tensor
    at::Tensor result = npu_preparation::apply_tensor_without_format(grad);
    result.copy_(grad);
    // 计算输出结果
    return {result, result};
}
}  // namespace op_api
