#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

// Adam 优化器 CUDA Kernel
__global__ void AdamAccumulateGradKernel(const float *grad_ptr, float *param_ptr, float *m_ptr, float *v_ptr,
                                          float learning_rate, float beta1, float beta2, float eps,
                                          float bias_correction1, float bias_correction2, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    const float g = grad_ptr[idx];
    
    // 更新一阶矩估计 (动量)
    m_ptr[idx] = beta1 * m_ptr[idx] + (1.0f - beta1) * g;
    
    // 更新二阶矩估计 (RMSprop)
    v_ptr[idx] = beta2 * v_ptr[idx] + (1.0f - beta2) * g * g;
    
    // 偏差修正
    const float m_hat = m_ptr[idx] / bias_correction1;
    const float v_hat = v_ptr[idx] / bias_correction2;
    
    // 更新参数
    param_ptr[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // 实现Adam优化器的梯度累积和参数更新（CUDA版本）
    // =================================== 作业 ===================================

    const size_t num_elements = grad->NumElements();
    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    // 计算偏差修正系数
    const float bias_correction1 = 1.0f - powf(beta1, static_cast<float>(t));
    const float bias_correction2 = 1.0f - powf(beta2, static_cast<float>(t));

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(
        grad_ptr, param_ptr, m_ptr, v_ptr, learning_rate, beta1, beta2, eps,
        bias_correction1, bias_correction2, num_elements);
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
