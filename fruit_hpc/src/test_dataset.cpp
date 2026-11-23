
#include "dataset.h"
#include <iostream>

int main() {

    std::map<std::string, int> class_to_label;

    auto records = load_image_paths(
        "/content/fruit_hpc/data/fruits360/Training",
        class_to_label
    );

    std::cout << "Total images: " << records.size() << "\n";
    std::cout << "Total classes: " << class_to_label.size() << "\n";

    FruitDataset dataset(records);

    // Load first image
    auto sample = dataset.get(0);

    std::cout << "\nImage tensor shape: "
              << sample.data.sizes() << "\n";

    std::cout << "Label: "
              << sample.target.item<int64_t>() << "\n";

    std::cout << "Minimum pixel value: "
              << sample.data.min().item<float>() << "\n";

    std::cout << "Maximum pixel value: "
              << sample.data.max().item<float>() << "\n";

    return 0;
}
