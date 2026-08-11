#ifndef TORCHATTACKS_PGD_KERNEL_CUH
#define TORCHATTACKS_PGD_KERNEL_CUH

#include <cuda_runtime.h>

namespace torchattacks {
namespace kernels {

void launch_fused_pgd_update(
    float* adv_images,
    const float* orig_images,
    const float* grads,
    int total_elements,
    float alpha,
    float eps,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace torchattacks

#endif // TORCHATTACKS_PGD_KERNEL_CUH