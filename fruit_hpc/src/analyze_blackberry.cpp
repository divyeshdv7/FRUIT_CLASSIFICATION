
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <filesystem>

int main() {

    const int N = 279;

    std::string train_path =
        "/content/fruit_hpc/data/fruits360/Training";

    std::vector<std::string> names;

    for (const auto& entry :
         std::filesystem::directory_iterator(train_path)) {

        if (entry.is_directory())
            names.push_back(entry.path().filename().string());
    }

    std::sort(names.begin(), names.end());

    std::ifstream file(
        "/content/fruit_hpc/results/confusion_matrix.csv"
    );

    if (!file.is_open()) {
        std::cout << "Cannot open confusion matrix!\n";
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

    std::cout << "\nBLACKBERRY CLASS ANALYSIS\n";

    for (int i = 0; i < N; i++) {

        if (names[i].find("Blackberry") == std::string::npos &&
            names[i].find("BlackBerry") == std::string::npos)
            continue;

        int total = 0;

        for (int j = 0; j < N; j++)
            total += matrix[i][j];

        std::cout << "\nActual class: " << names[i]
                  << " | Test images: " << total
                  << " | Correct: " << matrix[i][i] << "\n";

        std::vector<std::pair<int, int>> predictions;

        for (int j = 0; j < N; j++) {
            if (j != i && matrix[i][j] > 0)
                predictions.push_back({matrix[i][j], j});
        }

        std::sort(predictions.rbegin(), predictions.rend());

        std::cout << "Most frequent predictions:\n";

        for (int k = 0; k < std::min(3, (int)predictions.size()); k++) {
            int count = predictions[k].first;
            int predicted_id = predictions[k].second;

            std::cout << "  " << names[predicted_id]
                      << " : " << count << "\n";
        }
    }

    return 0;
}
