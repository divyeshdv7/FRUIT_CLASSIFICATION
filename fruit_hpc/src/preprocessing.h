
#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

struct ImageRecord {
    std::string path;
    int label;
};

std::vector<ImageRecord> load_image_paths(
    const std::string& root,
    std::map<std::string, int>& class_to_label
);

std::vector<cv::Mat> preprocess_sequential(
    const std::vector<ImageRecord>& records,
    int image_size
);

std::vector<cv::Mat> preprocess_openmp(
    const std::vector<ImageRecord>& records,
    int image_size
);

#endif
