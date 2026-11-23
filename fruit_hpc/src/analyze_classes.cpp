
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <filesystem>
#include <iomanip>

struct ClassResult {
    int id;
    std::string name;
    int total;
    int correct;
    double accuracy;
};

int main() {

    const int N = 279;

    std::string path =
        "/content/fruit_hpc/data/fruits360/Training";

    std::vector<std::string> names;

    for (const auto& entry :
         std::filesystem::directory_iterator(path)) {

        if (entry.is_directory())
            names.push_back(entry.path().filename().string());
    }

    std::sort(names.begin(), names.end());

    std::ifstream file(
        "/content/fruit_hpc/results/confusion_matrix.csv"
    );

    if (!file.is_open()) {
        std::cout << "Cannot open CSV file!\n";
        return 1;
    }

    std::vector<std::vector<int>> matrix(
        N, std::vector<int>(N)
    );

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            char comma;
            file >> matrix[i][j];

            if (j < N - 1)
                file >> comma;
        }
    }

    std::vector<ClassResult> results;

    double macro_accuracy = 0.0;
    int valid_classes = 0;

    for (int i = 0; i < N; i++) {

        int total = 0;
        int correct = matrix[i][i];

        for (int j = 0; j < N; j++)
            total += matrix[i][j];

        if (total == 0)
            continue;

        double accuracy =
            100.0 * correct / total;

        results.push_back({
            i, names[i], total, correct, accuracy
        });

        macro_accuracy += accuracy;
        valid_classes++;
    }

    // Best classes: highest accuracy first
    std::sort(
        results.begin(), results.end(),
        [](const ClassResult& a, const ClassResult& b) {
            return a.accuracy > b.accuracy;
        }
    );

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "\nTOP 10 BEST CLASSES\n";
    std::cout << "Fruit\t\tAccuracy\tCorrect/Total\n";

    for (int i = 0; i < std::min(10, (int)results.size()); i++) {
        auto r = results[i];

        std::cout << r.name << "\t"
                  << r.accuracy << "%\t"
                  << r.correct << "/" << r.total << "\n";
    }

    // Worst classes: lowest accuracy first
    std::sort(
        results.begin(), results.end(),
        [](const ClassResult& a, const ClassResult& b) {
            return a.accuracy < b.accuracy;
        }
    );

    std::cout << "\nTOP 10 WORST CLASSES\n";
    std::cout << "Fruit\t\tAccuracy\tCorrect/Total\n";

    for (int i = 0; i < std::min(10, (int)results.size()); i++) {
        auto r = results[i];

        std::cout << r.name << "\t"
                  << r.accuracy << "%\t"
                  << r.correct << "/" << r.total << "\n";
    }

    std::cout << "\nMacro Average Class Accuracy: "
              << macro_accuracy / valid_classes << "%\n";

    return 0;
}
