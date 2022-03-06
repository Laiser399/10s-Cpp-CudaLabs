cmake_minimum_required(VERSION 3.21)
project(Lab1_tests CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Lab1_tests main.cu)

set_target_properties(Lab1_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
