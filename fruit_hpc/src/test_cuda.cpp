
#include <iostream>
#include <vector>

extern "C" void normalize_on_gpu(
    const unsigned char* input,
    float* output,
    int size
);

int main() {

    const int size = 10;

    unsigned char input[size] = {
        0, 25, 50, 75, 100,
        125, 150, 175, 200, 255
    };

    std::vector<float> output(size);

    normalize_on_gpu(input, output.data(), size);

    std::cout << "CUDA normalization results:\n";

    for (int i = 0; i < size; i++) {
        std::cout << (int)input[i]
                  << " -> " << output[i] << std::endl;
    }

    return 0;
}
