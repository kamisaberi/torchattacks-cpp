#include "torchattacks/kernels/pgd_kernel.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace torchattacks {
namespace kernels {

__global__ void fused_pgd_update_kernel(
    float* __restrict__ adv_images,
    const float* __restrict__ orig_images,
    const float* __restrict__ grads,
    int total_elements,
    float alpha,
    float eps) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = grads[idx];
        float sign_g = (g > 0.0f) ? 1.0f : ((g < 0.0f) ? -1.0f : 0.0f);

        float adv = adv_images[idx] + alpha * sign_g;
        float orig = orig_images[idx];

        float eta = adv - orig;
        eta = fminf(fmaxf(eta, -eps), eps);

        adv_images[idx] = fminf(fmaxf(orig + eta, 0.0f), 1.0f);
    }
}

void launch_fused_pgd_update(
    float* adv_images,
    const float* orig_images,
    const float* grads,
    int total_elements,
    float alpha,
    float eps,
    cudaStream_t stream) 
{
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    fused_pgd_update_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        adv_images, orig_images, grads, total_elements, alpha, eps
    );
}

} // namespace kernels
} // namespace torchattacks