
#include "preprocessing.h"
#include <algorithm>

int main() {

    std::string dataset_path =
        "/content/fruit_hpc/data/fruits360/Training";

    std::map<std::string, int> class_to_label;

    // Step 1: Read image paths and labels
    auto records = load_image_paths(
        dataset_path,
        class_to_label
    );

    if (records.empty()) {
        std::cerr << "No images found!" << std::endl;
        return 1;
    }

    // Step 2: Test with only 100 images
    if (records.size() > 100) {
        records.resize(100);
    }

    // Step 3: Sequential preprocessing
    auto images = preprocess_sequential(records, 100);

    // Step 4: Display sample information
    if (!images.empty()) {

        std::cout << "\nFirst image information:" << std::endl;

        std::cout << "Rows: "
                  << images[0].rows << std::endl;

        std::cout << "Columns: "
                  << images[0].cols << std::endl;

        std::cout << "Channels: "
                  << images[0].channels() << std::endl;

        std::cout << "Data type: "
                  << images[0].type() << std::endl;

        std::cout << "First pixel (RGB): "
                  << images[0].at<cv::Vec3f>(0, 0)
                  << std::endl;
    }

    return 0;
}
