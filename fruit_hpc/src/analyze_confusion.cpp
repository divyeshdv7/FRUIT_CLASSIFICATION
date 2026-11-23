
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

struct Confusion {
    int actual;
    int predicted;
    int count;
};

int main() {

    const int N = 279;

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

    std::vector<Confusion> errors;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            if (i != j && matrix[i][j] > 0) {
                errors.push_back({i, j, matrix[i][j]});
            }
        }
    }

    std::sort(
        errors.begin(),
        errors.end(),
        [](const Confusion& a, const Confusion& b) {
            return a.count > b.count;
        }
    );

    std::cout << "\nTop 10 Most Confused Class Pairs\n";
    std::cout << "Actual\tPredicted\tCount\n";

    for (int i = 0; i < std::min(10, (int)errors.size()); i++) {
        std::cout << errors[i].actual << "\t"
                  << errors[i].predicted << "\t\t"
                  << errors[i].count << "\n";
    }

    return 0;
}
