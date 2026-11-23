
cmake_minimum_required(VERSION 3.18)

project(FruitHPC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(test_dataset
    src/test_dataset.cpp
    src/preprocessing.cpp
)

add_executable(train
    src/train.cpp
    src/preprocessing.cpp
)

add_executable(evaluate
    src/evaluate.cpp
    src/preprocessing.cpp
)

foreach(target test_dataset train evaluate)
    target_link_libraries(${target} PRIVATE
        "${TORCH_LIBRARIES}"
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
    )
    target_compile_options(${target} PRIVATE ${TORCH_CXX_FLAGS})
endforeach()
