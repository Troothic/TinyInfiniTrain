#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "Eigen/Dense"
#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== Assignment ===================================
    // CPU matrix multiplication forward: input[..., M, K] @ other[..., K, N] -> output[..., M, N]
    // =================================== Assignment ===================================

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);

    // Get matrix dimensions
    const int64_t M = input_dims[input_dims.size() - 2];
    const int64_t K = input_dims[input_dims.size() - 1];
    const int64_t N = other_dims[other_dims.size() - 1];
    
    // Verify dimension match
    CHECK_EQ(K, other_dims[other_dims.size() - 2]) << "Matrix dimension mismatch";

    // Compute batch dimensions
    const int64_t input_batch = std::accumulate(input_dims.begin(), input_dims.end() - 2, 1LL, std::multiplies<int64_t>{});
    const int64_t other_batch = std::accumulate(other_dims.begin(), other_dims.end() - 2, 1LL, std::multiplies<int64_t>{});
    CHECK(input_batch == other_batch || input_batch == 1 || other_batch == 1) << "Batch dimension incompatible";
    const int64_t batch = std::max(input_batch, other_batch);

    // Build output dimensions
    std::vector<int64_t> output_dims;
    if (input_dims.size() >= other_dims.size()) {
        output_dims = std::vector<int64_t>(input_dims.begin(), input_dims.end() - 2);
    } else {
        output_dims = std::vector<int64_t>(other_dims.begin(), other_dims.end() - 2);
    }
    output_dims.push_back(M);
    output_dims.push_back(N);

    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    // Get data pointers
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *other_ptr = static_cast<const float *>(other->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    // Execute batched matrix multiplication
    for (int64_t b = 0; b < batch; ++b) {
        const float *A = input_ptr + (input_batch == 1 ? 0 : b) * M * K;
        const float *B = other_ptr + (other_batch == 1 ? 0 : b) * K * N;
        float *C = output_ptr + b * M * N;

        // Use Eigen Map for matrix multiplication
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_mat(A, M, K);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_mat(B, K, N);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_mat(C, M, N);
        C_mat = A_mat * B_mat;
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== Assignment ===================================
    // CPU matrix multiplication backward:
    // grad_input = grad_output @ other^T
    // grad_other = input^T @ grad_output
    // =================================== Assignment ===================================

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &grad_dims = grad_output->Dims();

    // Get matrix dimensions
    const int64_t M = input_dims[input_dims.size() - 2];
    const int64_t K = input_dims[input_dims.size() - 1];
    const int64_t N = other_dims[other_dims.size() - 1];

    // Compute batch dimensions
    const int64_t input_batch = std::accumulate(input_dims.begin(), input_dims.end() - 2, 1LL, std::multiplies<int64_t>{});
    const int64_t other_batch = std::accumulate(other_dims.begin(), other_dims.end() - 2, 1LL, std::multiplies<int64_t>{});
    const int64_t batch = std::accumulate(grad_dims.begin(), grad_dims.end() - 2, 1LL, std::multiplies<int64_t>{});

    // Create gradient tensors
    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);

    // Get data pointers
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *other_ptr = static_cast<const float *>(other->DataPtr());
    const float *grad_out_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    float *grad_other_ptr = static_cast<float *>(grad_other->DataPtr());

    // Initialize to zero
    std::memset(grad_input_ptr, 0, grad_input->SizeInBytes());
    std::memset(grad_other_ptr, 0, grad_other->SizeInBytes());

    // Execute batched backward
    for (int64_t b = 0; b < batch; ++b) {
        const float *A = input_ptr + (input_batch == 1 ? 0 : b) * M * K;
        const float *B = other_ptr + (other_batch == 1 ? 0 : b) * K * N;
        const float *dC = grad_out_ptr + b * M * N;
        float *dA = grad_input_ptr + (input_batch == 1 ? 0 : b) * M * K;
        float *dB = grad_other_ptr + (other_batch == 1 ? 0 : b) * K * N;

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_mat(A, M, K);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_mat(B, K, N);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dC_mat(dC, M, N);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dA_mat(dA, M, K);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dB_mat(dB, K, N);

        // grad_input = grad_output @ other^T
        dA_mat += dC_mat * B_mat.transpose();
        // grad_other = input^T @ grad_output
        dB_mat += A_mat.transpose() * dC_mat;
    }

    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
