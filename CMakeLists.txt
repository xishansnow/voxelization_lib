cmake_minimum_required(VERSION 3.10)
project(voxelization_lib VERSION 1.0.0 LANGUAGES CXX CUDA)


# Set CUDA architectures
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)  # 60=Pascal, 61=GP102, 70=Volta, 75=Turing, 80=Ampere, 86=RTX30



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

# Set compiler flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")
    message(STATUS "Building in Debug mode with debug symbols")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    message(STATUS "Building in Release mode")
endif()

# Find CUDA using modern CMake
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(ENABLE_CUDA ON)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    # Add ENABLE_CUDA definition to compiler flags
    add_definitions(-DENABLE_CUDA)
else()
    set(ENABLE_CUDA OFF)
    message(STATUS "CUDA not found - GPU algorithms will use CPU fallback")
endif()

# Find OctoMap
find_package(octomap QUIET)
if(octomap_FOUND)
    set(ENABLE_OCTOMAP ON)
    message(STATUS "OctoMap found: ${octomap_VERSION}")
else()
    # Try pkg-config as fallback
    pkg_check_modules(OCTOMAP QUIET octomap)
    if(OCTOMAP_FOUND)
        set(ENABLE_OCTOMAP ON)
        message(STATUS "OctoMap found via pkg-config")
    else()
        set(ENABLE_OCTOMAP OFF)
        message(STATUS "OctoMap not found - .bt export will be disabled")
    endif()
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
    src/gpu_voxelization.cu
    src/gpu_voxelization_optimized.cu
    src/gpu_voxelization_optimized_impl.cpp
    src/forceflow_voxelizer.cu
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
    include/gpu_voxelization_optimized.hpp
    include/forceflow_voxelizer.hpp
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

# Add OctoMap support for test_voxelization
if(ENABLE_OCTOMAP)
    if(octomap_FOUND)
        target_link_libraries(test_voxelization octomap octomath)
        target_include_directories(test_voxelization PRIVATE ${octomap_INCLUDE_DIRS})
        target_compile_definitions(test_voxelization PRIVATE ENABLE_OCTOMAP)
    elseif(OCTOMAP_FOUND)
        target_link_libraries(test_voxelization ${OCTOMAP_LIBRARIES})
        target_include_directories(test_voxelization PRIVATE ${OCTOMAP_INCLUDE_DIRS})
        target_compile_options(test_voxelization PRIVATE ${OCTOMAP_CFLAGS_OTHER})
        target_compile_definitions(test_voxelization PRIVATE ENABLE_OCTOMAP)
    endif()
    message(STATUS "OctoMap enabled for test_voxelization")
endif()

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

# Create GPU optimization benchmark executable
add_executable(gpu_optimization_benchmark examples/gpu_optimization_benchmark.cpp)
target_link_libraries(gpu_optimization_benchmark voxelization_lib ${ASSIMP_LIBRARIES})

# Enable CUDA for GPU optimization benchmark
if(CUDAToolkit_FOUND)
    target_link_libraries(gpu_optimization_benchmark CUDA::cudart)
    set_target_properties(gpu_optimization_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    message(STATUS "CUDA enabled for GPU optimization benchmark")
endif()

# Enable C++17 for GPU optimization benchmark
target_compile_features(gpu_optimization_benchmark PRIVATE cxx_std_17)

# Print configuration info
message(STATUS "Voxelization Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen3: ${EIGEN3_INCLUDE_DIRS}")

# Create ForceFlow example executable
add_executable(forceflow_example examples/forceflow_example.cpp)
target_link_libraries(forceflow_example voxelization_lib ${ASSIMP_LIBRARIES})

# Enable CUDA for ForceFlow example
if(CUDAToolkit_FOUND)
    target_link_libraries(forceflow_example CUDA::cudart)
    set_target_properties(forceflow_example PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    message(STATUS "CUDA enabled for ForceFlow example")
endif()
