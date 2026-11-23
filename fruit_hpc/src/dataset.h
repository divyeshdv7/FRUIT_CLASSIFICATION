
#ifndef DATASET_H
#define DATASET_H

#include "preprocessing.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

struct FruitDataset : torch::data::datasets::Dataset<FruitDataset> {

    std::vector<ImageRecord> records;

    explicit FruitDataset(const std::vector<ImageRecord>& data) {
        records = data;
    }

    torch::data::Example<> get(size_t index) override {

        const auto& record = records.at(index);

        cv::Mat image = cv::imread(record.path);

        cv::resize(image, image, cv::Size(100, 100));

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        image.convertTo(image, CV_32FC3, 1.0 / 255.0);

        auto tensor = torch::from_blob(
            image.data,
            {100, 100, 3},
            torch::kFloat32
        ).clone();

        tensor = tensor.permute({2, 0, 1});

        auto label = torch::tensor(
            record.label,
            torch::kLong
        );

        return {tensor, label};
    }

    torch::optional<size_t> size() const override {
        return records.size();
    }
};

#endif
