cmake_minimum_required(VERSION 3.21)
project(Lab2_tests CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Lab2_tests main.cu)

set_target_properties(Lab2_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
