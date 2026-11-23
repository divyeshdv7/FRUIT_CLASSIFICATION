
#include "preprocessing.h"
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <omp.h>

namespace fs = std::filesystem;

std::vector<ImageRecord> load_image_paths(
    const std::string& root,
    std::map<std::string, int>& class_to_label
) {
    std::vector<ImageRecord> records;
    std::vector<fs::path> class_dirs;

    for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.is_directory()) {
            class_dirs.push_back(entry.path());
        }
    }

    std::sort(class_dirs.begin(), class_dirs.end());

    int label = 0;

    for (const auto& class_dir : class_dirs) {
        std::string class_name = class_dir.filename().string();
        class_to_label[class_name] = label;

        for (const auto& file : fs::directory_iterator(class_dir)) {
            if (file.is_regular_file() &&
                file.path().extension() == ".jpg") {

                records.push_back({
                    file.path().string(),
                    label
                });
            }
        }
        label++;
    }

    std::cout << "Classes found: "
              << class_to_label.size() << std::endl;

    std::cout << "Images found: "
              << records.size() << std::endl;

    return records;
}


// Sequential preprocessing
std::vector<cv::Mat> preprocess_sequential(
    const std::vector<ImageRecord>& records,
    int image_size
) {
    std::vector<cv::Mat> images;
    images.reserve(records.size());

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& record : records) {
        cv::Mat image = cv::imread(record.path);

        if (image.empty()) continue;

        cv::resize(image, image,
                   cv::Size(image_size, image_size));

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        image.convertTo(image, CV_32FC3, 1.0 / 255.0);

        images.push_back(image);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double elapsed =
        std::chrono::duration<double>(end - start).count();

    std::cout << "Sequential time: "
              << elapsed << " seconds" << std::endl;

    return images;
}


// OpenMP parallel preprocessing
std::vector<cv::Mat> preprocess_openmp(
    const std::vector<ImageRecord>& records,
    int image_size
) {
    // Allocate output slots before parallel execution.
    std::vector<cv::Mat> images(records.size());

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(records.size()); i++) {

        cv::Mat image = cv::imread(records[i].path);

        if (image.empty()) continue;

        cv::resize(image, image,
                   cv::Size(image_size, image_size));

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        image.convertTo(image, CV_32FC3, 1.0 / 255.0);

        // Each thread writes to a different vector element.
        images[i] = image;
    }

    auto end = std::chrono::high_resolution_clock::now();

    double elapsed =
        std::chrono::duration<double>(end - start).count();

    std::cout << "OpenMP time: "
              << elapsed << " seconds" << std::endl;

    std::cout << "OpenMP threads: "
              << omp_get_max_threads() << std::endl;

    return images;
}
