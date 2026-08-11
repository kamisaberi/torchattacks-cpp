#include "torchattacks/fgsm.h"

namespace torchattacks {

FGSM::FGSM(ModelFn model, float eps) 
    : Attack(std::move(model)), eps_(eps) {}

FGSM::FGSM(std::shared_ptr<torch::nn::Module> model, float eps) 
    : Attack(std::move(model)), eps_(eps) {}

torch::Tensor FGSM::forward(const torch::Tensor& images, const torch::Tensor& labels) {
    torch::Tensor adv_images = images.clone().detach().set_requires_grad(true);

    auto outputs = forward_model(adv_images);
    auto loss = torch::nn::functional::cross_entropy(outputs, labels);

    loss.backward();

    torch::NoGradGuard no_grad;
    torch::Tensor grad = adv_images.grad();
    adv_images = adv_images + eps_ * grad.sign();
    adv_images = torch::clamp(adv_images, 0.0f, 1.0f);

    return adv_images.detach();
}

} // namespace torchattacks