
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>

extern "C" void normalize_on_gpu(
    const unsigned char* input,
    float* output,
    int size
);

namespace fs = std::filesystem;

int main() {

    std::string root =
        "/content/fruit_hpc/data/fruits360/Training";

    std::vector<std::string> paths;

    for (const auto& dir : fs::directory_iterator(root)) {
        if (!dir.is_directory()) continue;

        for (const auto& file : fs::directory_iterator(dir.path())) {
            if (file.path().extension() == ".jpg") {
                paths.push_back(file.path().string());
            }
        }
    }

    std::sort(paths.begin(), paths.end());

    if (paths.size() < 500) {
        std::cerr << "Not enough images!" << std::endl;
        return 1;
    }

    paths.resize(500);

    std::vector<unsigned char> raw_pixels;

    // OpenCV: Load, resize and convert images
    for (const auto& path : paths) {

        cv::Mat image = cv::imread(path);

        if (image.empty()) continue;

        cv::resize(image, image, cv::Size(100, 100));

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        raw_pixels.insert(
            raw_pixels.end(),
            image.data,
            image.data + image.total() * image.elemSize()
        );
    }

    int size = raw_pixels.size();

    std::cout << "Images loaded: "
              << raw_pixels.size() / 30000 << std::endl;

    std::cout << "Total pixel values: "
              << size << std::endl;

    // CPU normalization
    std::vector<float> cpu_output(size);

    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < size; i++) {
        cpu_output[i] = raw_pixels[i] / 255.0f;
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double>(
            cpu_end - cpu_start
        ).count();

    // GPU normalization
    std::vector<float> gpu_output(size);

    auto gpu_start = std::chrono::high_resolution_clock::now();

    normalize_on_gpu(
        raw_pixels.data(),
        gpu_output.data(),
        size
    );

    auto gpu_end = std::chrono::high_resolution_clock::now();

    double gpu_time =
        std::chrono::duration<double>(
            gpu_end - gpu_start
        ).count();

    // Verify correctness
    float max_error = 0.0f;

    for (int i = 0; i < size; i++) {
        max_error = std::max(
            max_error,
            std::abs(cpu_output[i] - gpu_output[i])
        );
    }

    std::cout << "\nCPU normalization time: "
              << cpu_time << " seconds" << std::endl;

    std::cout << "CUDA normalization time: "
              << gpu_time << " seconds" << std::endl;

    std::cout << "Maximum difference: "
              << max_error << std::endl;

    if (max_error < 1e-6f) {
        std::cout << "CPU and GPU results match!" << std::endl;
    }

    return 0;
}
