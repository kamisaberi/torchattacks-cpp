#ifndef TORCHATTACKS_FGSM_H
#define TORCHATTACKS_FGSM_H

#include "torchattacks/attack.h"

namespace torchattacks {

class FGSM : public Attack {
public:
    FGSM(ModelFn model, float eps = 8.0f / 255.0f);
    FGSM(std::shared_ptr<torch::nn::Module> model, float eps = 8.0f / 255.0f);

    torch::Tensor forward(const torch::Tensor& images, 
                                  const torch::Tensor& labels) override;

private:
    float eps_;
};

} // namespace torchattacks

#endif // TORCHATTACKS_FGSM_H