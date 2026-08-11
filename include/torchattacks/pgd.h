#ifndef TORCHATTACKS_PGD_H
#define TORCHATTACKS_PGD_H

#include "torchattacks/attack.h"

namespace torchattacks {

class PGD : public Attack {
public:
    PGD(ModelFn model, float eps = 8.0f / 255.0f, float alpha = 2.0f / 255.0f, int steps = 10, bool random_start = true);
    PGD(std::shared_ptr<torch::nn::Module> model, float eps = 8.0f / 255.0f, float alpha = 2.0f / 255.0f, int steps = 10, bool random_start = true);

    torch::Tensor forward(const torch::Tensor& images, 
                                  const torch::Tensor& labels) override;

private:
    float eps_;
    float alpha_;
    int steps_;
    bool random_start_;
};

} // namespace torchattacks

#endif // TORCHATTACKS_PGD_H