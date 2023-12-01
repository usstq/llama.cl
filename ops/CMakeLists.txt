cmake_minimum_required(VERSION 3.4...3.18)

project(llmops)

message("====== CMAKE_CXX_COMPILER  = ${CMAKE_CXX_COMPILER}")
message("====== CMAKE_CXX_COMPILER_ID   = ${CMAKE_CXX_COMPILER_ID}")

# manually add entry PYTHON_EXECUTABLE pointting to python38.exe

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (UNIX)
if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message("========== UNIX IntelLLVM ===========")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xhost")
    set (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -fsycl")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()
endif (UNIX)

if (MSVC)
if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # DPC++ related flags
    message("========== MSVC IntelLLVM ===========")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qxhost")
    set (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -fsycl")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch:AVX2")
endif ()
endif (MSVC)

add_subdirectory(pybind11)

pybind11_add_module(llmops main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(ops PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
