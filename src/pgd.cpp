#include "torchattacks/pgd.h"
#include "torchattacks/kernels/pgd_kernel.cuh"
#include <c10/cuda/CUDAStream.h>

namespace torchattacks {

PGD::PGD(ModelFn model, float eps, float alpha, int steps, bool random_start)
    : Attack(std::move(model)), eps_(eps), alpha_(alpha), steps_(steps), random_start_(random_start) {}

PGD::PGD(std::shared_ptr<torch::nn::Module> model, float eps, float alpha, int steps, bool random_start)
    : Attack(std::move(model)), eps_(eps), alpha_(alpha), steps_(steps), random_start_(random_start) {}

torch::Tensor PGD::forward(const torch::Tensor& images, const torch::Tensor& labels) {
    torch::Tensor orig_images = images.clone().detach();
    torch::Tensor adv_images = images.clone().detach();

    if (random_start_) {
        adv_images = adv_images + torch::empty_like(adv_images).uniform_(-eps_, eps_);
        adv_images = torch::clamp(adv_images, 0.0f, 1.0f);
    }

    int total_elements = images.numel();

    for (int i = 0; i < steps_; ++i) {
        adv_images.requires_grad_(true);

        auto outputs = forward_model(adv_images);
        auto loss = torch::nn::functional::cross_entropy(outputs, labels);

        loss.backward();

        torch::Tensor grad = adv_images.grad();

        {
            torch::NoGradGuard no_grad;
            if (adv_images.is_cuda()) {
                cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
                kernels::launch_fused_pgd_update(
                    adv_images.data_ptr<float>(),
                    orig_images.data_ptr<float>(),
                    grad.data_ptr<float>(),
                    total_elements,
                    alpha_,
                    eps_,
                    stream
                );
            } else {
                // CPU fallback execution path
                torch::Tensor grad_sign = grad.sign();
                adv_images = adv_images + alpha_ * grad_sign;
                torch::Tensor eta = torch::clamp(adv_images - orig_images, -eps_, eps_);
                adv_images = torch::clamp(orig_images + eta, 0.0f, 1.0f);
            }
        }
    }

    return adv_images.detach();
}

} // namespace torchattacks