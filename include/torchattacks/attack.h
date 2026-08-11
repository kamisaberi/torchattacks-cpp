#ifndef TORCHATTACKS_ATTACK_H
#define TORCHATTACKS_ATTACK_H

#include <torch/torch.h>
#include <memory>
#include <functional>

namespace torchattacks {

// Flexible wrapper allowing both torch::nn::Module and custom forward functions
using ModelFn = std::function<torch::Tensor(const torch::Tensor&)>;

class Attack {
public:
    explicit Attack(ModelFn model);
    explicit Attack(std::shared_ptr<torch::nn::Module> model);
    virtual ~Attack() = default;

    // Core execution interface
    virtual torch::Tensor forward(const torch::Tensor& images, 
                                  const torch::Tensor& labels) = 0;

protected:
    torch::Tensor forward_model(const torch::Tensor& inputs);

    ModelFn model_fn_;
    std::shared_ptr<torch::nn::Module> model_ptr_;
};

} // namespace torchattacks

#endif // TORCHATTACKS_ATTACK_H