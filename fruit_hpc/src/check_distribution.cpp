
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <string>

int main() {

    std::string path =
        "/content/fruit_hpc/data/fruits360/Training";

    std::vector<std::string> classes;

    for (const auto& entry :
         std::filesystem::directory_iterator(path)) {

        if (entry.is_directory())
            classes.push_back(entry.path().filename().string());
    }

    std::sort(classes.begin(), classes.end());

    std::cout << "BLACKBERRY TRAINING DISTRIBUTION\n\n";

    for (const auto& name : classes) {

        if (name.find("Blackberry") == std::string::npos &&
            name.find("BlackBerry") == std::string::npos)
            continue;

        std::string class_path = path + "/" + name;

        int count = 0;

        for (const auto& image :
             std::filesystem::directory_iterator(class_path)) {

            if (image.is_regular_file())
                count++;
        }

        std::cout << name << " : "
                  << count << " training images\n";
    }

    return 0;
}
