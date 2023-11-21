cmake_minimum_required(VERSION 3.4...3.18)
project(llmops)

# manually add entry PYTHON_EXECUTABLE pointting to python38.exe

set(PYTHON_EXECUTABLE "C:/Users/tingqian/AppData/Local/Programs/Python/Python38/python38.exe")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(pybind11)

pybind11_add_module(llmops main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(ops PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
