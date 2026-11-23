
#include "model.h"
#include "dataset.h"

#include <iostream>
#include <algorithm>
#include <vector>

int main() {

    torch::Device device(
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU
    );

    std::cout << "Evaluation device: " << device << "\n";

    // Load test dataset
    std::map<std::string, int> class_to_label;

    auto records = load_image_paths(
        "/content/fruit_hpc/data/fruits360/Test",
        class_to_label
    );

    std::cout << "Test images: " << records.size() << "\n";

    // Load trained model
    FruitCNN model;

    torch::load(model, "/content/fruit_hpc/results/fruit_model.pt");

    model->to(device);
    model->eval();

    const int batch_size = 32;

    int correct = 0;
    int total = 0;

    double total_loss = 0.0;
    int batches = 0;

    torch::nn::CrossEntropyLoss criterion;

    auto start = std::chrono::high_resolution_clock::now();

    // Evaluate in batches
    torch::NoGradGuard no_grad;

    for (size_t i = 0; i < records.size(); i += batch_size) {

        std::vector<torch::Tensor> images;
        std::vector<torch::Tensor> labels;

        size_t end = std::min(
            i + batch_size,
            records.size()
        );

        for (size_t j = i; j < end; j++) {

            FruitDataset dataset({records[j]});

            auto sample = dataset.get(0);

            images.push_back(sample.data);
            labels.push_back(sample.target);
        }

        auto input = torch::stack(images).to(device);
        auto target = torch::stack(labels).to(device);

        auto output = model->forward(input);

        auto loss = criterion(output, target);

        total_loss += loss.item<double>();

        auto prediction = output.argmax(1);

        correct += prediction.eq(target).sum().item<int>();

        total += target.size(0);

        batches++;
    }

    auto stop = std::chrono::high_resolution_clock::now();

    double elapsed =
        std::chrono::duration<double>(stop - start).count();

    std::cout << "\nTest Accuracy: "
              << (100.0 * correct / total) << "%\n";

    std::cout << "Average Test Loss: "
              << total_loss / batches << "\n";

    std::cout << "Correct predictions: "
              << correct << "/" << total << "\n";

    std::cout << "Evaluation time: "
              << elapsed << " seconds\n";

    return 0;
}
