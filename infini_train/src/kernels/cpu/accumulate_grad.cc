#include <cmath>
#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== Assignment ===================================
    // Adam optimizer gradient accumulation and parameter update
    // Adam algorithm:
    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    // m_hat = m_t / (1 - beta1^t)
    // v_hat = v_t / (1 - beta2^t)
    // theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
    // =================================== Assignment ===================================

    const int64_t num_elements = grad->NumElements();
    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    // Compute bias correction coefficients
    const float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(t));

    for (int64_t i = 0; i < num_elements; ++i) {
        const float g = grad_ptr[i];
        
        // Update first moment estimate (momentum)
        m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g;
        
        // Update second moment estimate (RMSprop)
        v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;
        
        // Bias correction
        const float m_hat = m_ptr[i] / bias_correction1;
        const float v_hat = v_ptr[i] / bias_correction2;
        
        // Update parameter
        param_ptr[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
