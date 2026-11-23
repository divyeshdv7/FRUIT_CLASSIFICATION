
#include "preprocessing.h"
#include <omp.h>

int main() {

    std::string path =
        "/content/fruit_hpc/data/fruits360/Training";

    std::map<std::string, int> class_to_label;

    auto records = load_image_paths(path, class_to_label);

    // Use the same 500 images for both tests.
    records.resize(500);

    std::cout << "\n--- Sequential Processing ---\n";

    auto sequential =
        preprocess_sequential(records, 100);

    std::cout << "Sequential images: "
              << sequential.size() << std::endl;

    std::cout << "\n--- OpenMP Processing ---\n";

    auto parallel =
        preprocess_openmp(records, 100);

    int successful = 0;

    for (const auto& image : parallel) {
        if (!image.empty()) successful++;
    }

    std::cout << "OpenMP images: "
              << successful << std::endl;

    std::cout << "\nBenchmark completed!" << std::endl;

    return 0;
}
