cmake_minimum_required(VERSION 3.10)
project(voxelization_lib VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find CUDA using modern CMake
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(ENABLE_CUDA ON)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
else()
    set(ENABLE_CUDA OFF)
    message(STATUS "CUDA not found - GPU algorithms will use CPU fallback")
endif()

# Include directories
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ASSIMP REQUIRED assimp)

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${ASSIMP_INCLUDE_DIRS})

if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Source files
set(SOURCES
    src/spatial_entities.cpp
    src/cpu_voxelization.cpp
    src/gpu_voxelization.cpp
    src/hybrid_voxelization.cpp
    src/voxelization_factory.cpp
    src/mesh_loader.cpp
)

# Header files
set(HEADERS
    include/voxelization_base.hpp
    include/spatial_entities.hpp
    include/voxelization_algorithms.hpp
    include/cpu_voxelization.hpp
    include/gpu_voxelization.hpp
    include/hybrid_voxelization.hpp
    include/voxelization_factory.hpp
    include/mesh_loader.hpp
    # include/simple_vector.hpp
)

# Create shared library
add_library(voxelization_lib SHARED ${SOURCES} ${HEADERS})

# Set library properties
set_target_properties(voxelization_lib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME voxelization
)

# Set compiler flags
target_compile_options(voxelization_lib PRIVATE
    -Wall
    -Wextra
    -O3
    -fPIC
)

# Set include directories for the library
target_include_directories(voxelization_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link libraries
target_link_libraries(voxelization_lib ${ASSIMP_LIBRARIES})

if(CUDAToolkit_FOUND)
    target_link_libraries(voxelization_lib CUDA::cudart)
    set_target_properties(voxelization_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Enable OpenMP for the main library
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(voxelization_lib OpenMP::OpenMP_CXX)
    target_compile_options(voxelization_lib PRIVATE ${OpenMP_CXX_FLAGS})
    # Also link the OpenMP runtime library explicitly
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_libraries(voxelization_lib gomp)
    endif()
    message(STATUS "OpenMP found and enabled for main library")
else()
    message(STATUS "OpenMP not found - parallel algorithms may not be available")
endif()

# Create test executable
add_executable(test_voxelization test/test_voxelization.cpp)
target_link_libraries(test_voxelization voxelization_lib Eigen3::Eigen ${ASSIMP_LIBRARIES})

# Create example executable
add_executable(example_usage examples/example_usage.cpp)
target_link_libraries(example_usage voxelization_lib ${ASSIMP_LIBRARIES})

# Create mesh voxelization example executable
add_executable(mesh_voxelization_example examples/mesh_voxelization_example.cpp)
target_link_libraries(mesh_voxelization_example voxelization_lib ${ASSIMP_LIBRARIES})

# Create performance benchmark executable
add_executable(performance_benchmark examples/performance_benchmark.cpp)
target_link_libraries(performance_benchmark voxelization_lib ${ASSIMP_LIBRARIES})

# Enable OpenMP for performance benchmark
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(performance_benchmark OpenMP::OpenMP_CXX)
    target_compile_options(performance_benchmark PRIVATE ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP found and enabled for performance benchmark")
else()
    message(STATUS "OpenMP not found - parallel algorithms may not be available")
endif()

# Create advanced performance benchmark executable
add_executable(advanced_benchmark examples/advanced_benchmark.cpp)
target_link_libraries(advanced_benchmark voxelization_lib ${ASSIMP_LIBRARIES})

# Enable OpenMP for advanced performance benchmark
if(OpenMP_CXX_FOUND)
    target_link_libraries(advanced_benchmark OpenMP::OpenMP_CXX)
    target_compile_options(advanced_benchmark PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Print configuration info
message(STATUS "Voxelization Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen3: ${EIGEN3_INCLUDE_DIRS}")
