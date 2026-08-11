#include "torchattacks/attack.h"

namespace torchattacks {

Attack::Attack(ModelFn model) 
    : model_fn_(std::move(model)), model_ptr_(nullptr) {}

Attack::Attack(std::shared_ptr<torch::nn::Module> model) 
    : model_ptr_(std::move(model)) {
    if (model_ptr_) {
        model_ptr_->eval();
    }
}

torch::Tensor Attack::forward_model(const torch::Tensor& inputs) {
    if (model_fn_) {
        return model_fn_(inputs);
    }
    if (model_ptr_) {
        return model_ptr_->forward({inputs}).toTensor();
    }
    TORCH_CHECK(false, "torchattacks-cpp: Model wrapper is uninitialized!");
}

} // namespace torchattacks