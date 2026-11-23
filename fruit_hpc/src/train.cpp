
#include "model.h"
#include "dataset.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <map>
#include <filesystem>

int main() {

    torch::Device device(
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU
    );

    std::cout << "Training device: " << device << "\n";

    // 1. Load training image paths
    std::map<std::string, int> class_to_label;

    auto all_records = load_image_paths(
        "/content/fruit_hpc/data/fruits360/Training",
        class_to_label
    );

    // 2. Group images by class
    std::map<int, std::vector<ImageRecord>> class_images;

    for (const auto& record : all_records) {
        class_images[record.label].push_back(record);
    }

    std::mt19937 rng(42);

    std::vector<ImageRecord> train_records;
    std::vector<ImageRecord> val_records;

    const int images_per_class = 20;

    // 3. Create balanced training and validation subsets
    for (auto& entry : class_images) {

        auto& images = entry.second;

        std::shuffle(images.begin(), images.end(), rng);

        int count = std::min(
            images_per_class,
            static_cast<int>(images.size())
        );

        int train_count = static_cast<int>(count * 0.8);

        for (int i = 0; i < count; i++) {

            if (i < train_count) {
                train_records.push_back(images[i]);
            }
            else {
                val_records.push_back(images[i]);
            }
        }
    }

    std::shuffle(train_records.begin(), train_records.end(), rng);
    std::shuffle(val_records.begin(), val_records.end(), rng);

    std::cout << "Classes: " << class_images.size() << "\n";
    std::cout << "Training images: " << train_records.size() << "\n";
    std::cout << "Validation images: " << val_records.size() << "\n";

    // 4. Create datasets
    FruitDataset train_dataset(train_records);
    FruitDataset val_dataset(val_records);

    // 5. Initialize CNN
    FruitCNN model;
    model->to(device);

    // 6. Loss function and optimizer
    torch::nn::CrossEntropyLoss criterion;

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(0.001)
    );

    const int batch_size = 32;
    const int epochs = 5;

    double best_val_accuracy = -1.0;

    std::filesystem::create_directories(
        "/content/fruit_hpc/results"
    );

    std::cout << "\nStarting training...\n";

    // 7. Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {

        model->train();

        double train_loss = 0.0;
        int train_correct = 0;
        int train_total = 0;
        int train_batches = 0;

        // Shuffle training order each epoch
        std::vector<size_t> order(train_records.size());

        for (size_t i = 0; i < order.size(); i++) {
            order[i] = i;
        }

        std::shuffle(order.begin(), order.end(), rng);

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < order.size(); i += batch_size) {

            std::vector<torch::Tensor> images;
            std::vector<torch::Tensor> labels;

            size_t end = std::min(
                i + batch_size,
                order.size()
            );

            for (size_t j = i; j < end; j++) {

                auto sample = train_dataset.get(order[j]);

                images.push_back(sample.data);
                labels.push_back(sample.target);
            }

            auto input = torch::stack(images).to(device);
            auto target = torch::stack(labels).to(device);

            optimizer.zero_grad();

            auto output = model->forward(input);

            auto loss = criterion(output, target);

            loss.backward();

            optimizer.step();

            train_loss += loss.item<double>();

            auto prediction = output.argmax(1);

            train_correct +=
                prediction.eq(target).sum().item<int>();

            train_total += target.size(0);

            train_batches++;
        }

        // 8. Validation
        model->eval();

        int val_correct = 0;
        int val_total = 0;
        double val_loss = 0.0;
        int val_batches = 0;

        {
            torch::NoGradGuard no_grad;

            for (size_t i = 0; i < val_records.size(); i += batch_size) {

                std::vector<torch::Tensor> images;
                std::vector<torch::Tensor> labels;

                size_t end = std::min(
                    i + batch_size,
                    val_records.size()
                );

                for (size_t j = i; j < end; j++) {

                    auto sample = val_dataset.get(j);

                    images.push_back(sample.data);
                    labels.push_back(sample.target);
                }

                auto input = torch::stack(images).to(device);
                auto target = torch::stack(labels).to(device);

                auto output = model->forward(input);

                auto loss = criterion(output, target);

                val_loss += loss.item<double>();

                auto prediction = output.argmax(1);

                val_correct +=
                    prediction.eq(target).sum().item<int>();

                val_total += target.size(0);

                val_batches++;
            }
        }

        // 9. Calculate metrics
        double train_accuracy =
            100.0 * train_correct / train_total;

        double validation_accuracy =
            100.0 * val_correct / val_total;

        auto stop = std::chrono::high_resolution_clock::now();

        double elapsed =
            std::chrono::duration<double>(stop - start).count();

        std::cout << "\nEpoch " << epoch + 1 << "/" << epochs << "\n";

        std::cout << "Training Loss: "
                  << train_loss / train_batches << "\n";

        std::cout << "Training Accuracy: "
                  << train_accuracy << "%\n";

        std::cout << "Validation Loss: "
                  << val_loss / val_batches << "\n";

        std::cout << "Validation Accuracy: "
                  << validation_accuracy << "%\n";

        std::cout << "Time: " << elapsed << " seconds\n";

        // 10. Save best model
        if (validation_accuracy > best_val_accuracy) {

            best_val_accuracy = validation_accuracy;

            torch::save(
                model,
                "/content/fruit_hpc/results/fruit_model.pt"
            );

            std::cout << "Best model checkpoint saved!\n";
        }
    }

    std::cout << "\nTraining completed!\n";

    std::cout << "Best validation accuracy: "
              << best_val_accuracy << "%\n";

    return 0;
}
