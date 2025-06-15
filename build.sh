#!/bin/bash

echo "=== Voxelization Library Build Script ==="

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Debug

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed!"
    exit 1
fi

# Build
echo "Building..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi

echo "Build completed successfully!"
echo "Test executable: ./test_voxelization"
echo "Example executable: ./example_usage"
echo "Library: libvoxelization.so" 
