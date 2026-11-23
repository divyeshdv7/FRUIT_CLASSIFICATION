
#include <iostream>
#include <torch/torch.h>

int main() {
    std::cout << "LibTorch is working!" << std::endl;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available through LibTorch!" << std::endl;

        auto tensor = torch::ones(
            {2, 3},
            torch::TensorOptions().device(torch::kCUDA)
        );

        std::cout << "GPU Tensor:" << std::endl;
        std::cout << tensor << std::endl;
    }
    else {
        std::cout << "CUDA is NOT available through LibTorch." << std::endl;
    }

    return 0;
}
