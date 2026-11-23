
#include "model.h"
#include <iostream>

int main() {

    // Select GPU if available
    torch::Device device(
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU
    );

    // Create CNN
    FruitCNN model;

    // Move model to GPU
    model->to(device);

    // Create 4 random RGB images (batch size = 4)
    auto input = torch::rand({4, 3, 100, 100}).to(device);

    // Perform forward propagation
    auto output = model->forward(input);

    std::cout << "CNN model initialized successfully!\n";
    std::cout << "Device: " << device << "\n";
    std::cout << "Input shape: " << input.sizes() << "\n";
    std::cout << "Output shape: " << output.sizes() << "\n";

    return 0;
}
