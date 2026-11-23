
#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>

struct FruitCNNImpl : torch::nn::Module {

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    FruitCNNImpl() {

        conv1 = register_module("conv1", torch::nn::Conv2d(3, 32, 3));
        conv2 = register_module("conv2", torch::nn::Conv2d(32, 64, 3));

        fc1 = register_module("fc1", torch::nn::Linear(64 * 23 * 23, 128));

        // 279 fruit classes
        fc2 = register_module("fc2", torch::nn::Linear(128, 279));
    }

    torch::Tensor forward(torch::Tensor x) {

        x = torch::relu(conv1->forward(x));
        x = torch::max_pool2d(x, 2);

        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, 2);

        x = x.view({x.size(0), -1});

        x = torch::relu(fc1->forward(x));

        return fc2->forward(x);
    }
};

TORCH_MODULE(FruitCNN);

#endif
