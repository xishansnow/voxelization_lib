#include "gpu_voxelization.hpp"
#include "cpu_voxelization.hpp"
#include "spatial_entities.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for box voxelization
__global__ void voxelizeBoxKernel(
    unsigned char* voxel_grid,
    int grid_x, int grid_y, int grid_z,
    double resolution_xy, double resolution_z,
    double origin_x, double origin_y, double origin_z,
    double center_x, double center_y, double center_z,
    double size_x, double size_y, double size_z,
    unsigned char cost_value) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid_x || y >= grid_y || z >= grid_z) {
        return;
    }

    // Convert voxel coordinates to world coordinates
    double world_x = origin_x + (x + 0.5) * resolution_xy;
    double world_y = origin_y + (y + 0.5) * resolution_xy;
    double world_z = origin_z + (z + 0.5) * resolution_z;

    // Check if point is inside box
    if (world_x >= center_x - size_x / 2 && world_x <= center_x + size_x / 2 &&
        world_y >= center_y - size_y / 2 && world_y <= center_y + size_y / 2 &&
        world_z >= center_z - size_z / 2 && world_z <= center_z + size_z / 2) {

        int index = z * grid_x * grid_y + y * grid_x + x;
        voxel_grid[index] = cost_value;
    }
}

// CUDA kernel for sphere voxelization
__global__ void voxelizeSphereKernel(
    unsigned char* voxel_grid,
    int grid_x, int grid_y, int grid_z,
    double resolution_xy, double resolution_z,
    double origin_x, double origin_y, double origin_z,
    double center_x, double center_y, double center_z,
    double radius,
    unsigned char cost_value) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid_x || y >= grid_y || z >= grid_z) {
        return;
    }

    // Convert voxel coordinates to world coordinates
    double world_x = origin_x + (x + 0.5) * resolution_xy;
    double world_y = origin_y + (y + 0.5) * resolution_xy;
    double world_z = origin_z + (z + 0.5) * resolution_z;

    // Check if point is inside sphere
    double dx = world_x - center_x;
    double dy = world_y - center_y;
    double dz = world_z - center_z;
    double distance_squared = dx * dx + dy * dy + dz * dz;

    if (distance_squared <= radius * radius) {
        int index = z * grid_x * grid_y + y * grid_x + x;
        voxel_grid[index] = cost_value;
    }
}

// CUDA kernel for cylinder voxelization
__global__ void voxelizeCylinderKernel(
    unsigned char* voxel_grid,
    int grid_x, int grid_y, int grid_z,
    double resolution_xy, double resolution_z,
    double origin_x, double origin_y, double origin_z,
    double center_x, double center_y, double center_z,
    double radius, double height,
    unsigned char cost_value) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid_x || y >= grid_y || z >= grid_z) {
        return;
    }

    // Convert voxel coordinates to world coordinates
    double world_x = origin_x + (x + 0.5) * resolution_xy;
    double world_y = origin_y + (y + 0.5) * resolution_xy;
    double world_z = origin_z + (z + 0.5) * resolution_z;

    // Check if point is inside cylinder
    double dx = world_x - center_x;
    double dy = world_y - center_y;
    double distance_squared = dx * dx + dy * dy;

    if (distance_squared <= radius * radius &&
        world_z >= center_z - height / 2 && world_z <= center_z + height / 2) {
        int index = z * grid_x * grid_y + y * grid_x + x;
        voxel_grid[index] = cost_value;
    }
}

namespace voxelization {

    // GPUCudaVoxelization implementation
    GPUCudaVoxelization::GPUCudaVoxelization()
        : grid_x_(0), grid_y_(0), grid_z_(0), resolution_xy_(0.0), resolution_z_(0.0),
        origin_x_(0.0), origin_y_(0.0), origin_z_(0.0), d_voxel_grid_(nullptr),
        grid_size_(0), gpu_available_(false) {

        // Check CUDA availability
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);

        if (error == cudaSuccess && device_count > 0) {
            gpu_available_ = true;
            cudaSetDevice(0); // Use first GPU

            // Get GPU info
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "CUDA GPU acceleration enabled: " << prop.name
                << " with " << prop.multiProcessorCount << " SMs" << std::endl;
        }
        else {
            gpu_available_ = false;
            std::cout << "CUDA GPU not available, falling back to CPU implementation" << std::endl;
        }
    }

    GPUCudaVoxelization::~GPUCudaVoxelization() {
        freeGPUMemory();
    }

    void GPUCudaVoxelization::initialize(int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z) {
        grid_x_ = grid_x;
        grid_y_ = grid_y;
        grid_z_ = grid_z;
        resolution_xy_ = resolution_xy;
        resolution_z_ = resolution_z;
        origin_x_ = origin_x;
        origin_y_ = origin_y;
        origin_z_ = origin_z;

        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_, 0);
        grid_size_ = voxel_grid_.size();

        if (gpu_available_) {
            allocateGPUMemory();
        }
    }

    void GPUCudaVoxelization::allocateGPUMemory() {
        if (d_voxel_grid_ != nullptr) {
            cudaFree(d_voxel_grid_);
        }

        cudaError_t error = cudaMalloc(&d_voxel_grid_, grid_size_ * sizeof(unsigned char));
        if (error != cudaSuccess) {
            std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(error) << std::endl;
            gpu_available_ = false;
        }
    }

    void GPUCudaVoxelization::freeGPUMemory() {
        if (d_voxel_grid_ != nullptr) {
            cudaFree(d_voxel_grid_);
            d_voxel_grid_ = nullptr;
        }
    }

    void GPUCudaVoxelization::copyToGPU() {
        if (gpu_available_ && d_voxel_grid_ != nullptr) {
            cudaError_t error = cudaMemcpy(d_voxel_grid_, voxel_grid_.data(),
                grid_size_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                std::cerr << "CUDA copy to device failed: " << cudaGetErrorString(error) << std::endl;
            }
        }
    }

    void GPUCudaVoxelization::copyFromGPU() {
        if (gpu_available_ && d_voxel_grid_ != nullptr) {
            cudaError_t error = cudaMemcpy(voxel_grid_.data(), d_voxel_grid_,
                grid_size_ * sizeof(unsigned char), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                std::cerr << "CUDA copy from device failed: " << cudaGetErrorString(error) << std::endl;
            }
        }
    }

    int GPUCudaVoxelization::voxelizeEntity(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {

        if (!gpu_available_) {
            // Fallback to CPU implementation
            CPUSequentialVoxelization cpu_voxelizer;
            cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);
            return cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
        }

        // Copy current voxel grid to GPU
        copyToGPU();

        int marked_voxels = 0;

        // Determine entity type and call appropriate GPU kernel
        if (auto box_entity = std::dynamic_pointer_cast<BoxEntity>(entity)) {
            marked_voxels = voxelizeBoxGPU(entity, buffer_size, cost_value);
        }
        else if (auto sphere_entity = std::dynamic_pointer_cast<SphereEntity>(entity)) {
            marked_voxels = voxelizeSphereGPU(entity, buffer_size, cost_value);
        }
        else if (auto cylinder_entity = std::dynamic_pointer_cast<CylinderEntity>(entity)) {
            marked_voxels = voxelizeCylinderGPU(entity, buffer_size, cost_value);
        }
        else {
            std::cerr << "Unknown entity type for GPU voxelization" << std::endl;
            return 0;
        }

        // Copy result back from GPU
        copyFromGPU();

        return marked_voxels;
    }

    int GPUCudaVoxelization::voxelizeBoxGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {

        auto box_entity = std::dynamic_pointer_cast<BoxEntity>(entity);
        if (!box_entity) return 0;

        // Get box parameters from properties
        auto properties = box_entity->getProperties();
        double center_x = properties["center_x"];
        double center_y = properties["center_y"];
        double center_z = properties["center_z"];
        double size_x = properties["size_x"];
        double size_y = properties["size_y"];
        double size_z = properties["size_z"];

        // Define block and grid dimensions
        dim3 block_size(16, 16, 4);
        dim3 grid_size(
            (grid_x_ + block_size.x - 1) / block_size.x,
            (grid_y_ + block_size.y - 1) / block_size.y,
            (grid_z_ + block_size.z - 1) / block_size.z
        );

        // Launch kernel
        voxelizeBoxKernel << <grid_size, block_size >> > (
            d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, size_x, size_y, size_z, cost_value);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
            return 0;
        }

        // Synchronize
        cudaDeviceSynchronize();

        // Count marked voxels (simplified - in practice you might want to do this on GPU)
        int marked_voxels = 0;
        copyFromGPU();
        for (size_t i = 0; i < voxel_grid_.size(); ++i) {
            if (voxel_grid_[i] == cost_value) {
                marked_voxels++;
            }
        }

        return marked_voxels;
    }

    int GPUCudaVoxelization::voxelizeSphereGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {

        auto sphere_entity = std::dynamic_pointer_cast<SphereEntity>(entity);
        if (!sphere_entity) return 0;

        // Get sphere parameters from properties
        auto properties = sphere_entity->getProperties();
        double center_x = properties["center_x"];
        double center_y = properties["center_y"];
        double center_z = properties["center_z"];
        double radius = properties["radius"];

        // Define block and grid dimensions
        dim3 block_size(16, 16, 4);
        dim3 grid_size(
            (grid_x_ + block_size.x - 1) / block_size.x,
            (grid_y_ + block_size.y - 1) / block_size.y,
            (grid_z_ + block_size.z - 1) / block_size.z
        );

        // Launch kernel
        voxelizeSphereKernel << <grid_size, block_size >> > (
            d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, radius, cost_value);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
            return 0;
        }

        // Synchronize
        cudaDeviceSynchronize();

        // Count marked voxels
        int marked_voxels = 0;
        copyFromGPU();
        for (size_t i = 0; i < voxel_grid_.size(); ++i) {
            if (voxel_grid_[i] == cost_value) {
                marked_voxels++;
            }
        }

        return marked_voxels;
    }

    int GPUCudaVoxelization::voxelizeCylinderGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {

        auto cylinder_entity = std::dynamic_pointer_cast<CylinderEntity>(entity);
        if (!cylinder_entity) return 0;

        // Get cylinder parameters from properties
        auto properties = cylinder_entity->getProperties();
        double center_x = properties["center_x"];
        double center_y = properties["center_y"];
        double center_z = properties["center_z"];
        double radius = properties["radius"];
        double height = properties["height"];

        // Define block and grid dimensions
        dim3 block_size(16, 16, 4);
        dim3 grid_size(
            (grid_x_ + block_size.x - 1) / block_size.x,
            (grid_y_ + block_size.y - 1) / block_size.y,
            (grid_z_ + block_size.z - 1) / block_size.z
        );

        // Launch kernel
        voxelizeCylinderKernel << <grid_size, block_size >> > (
            d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, radius, height, cost_value);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
            return 0;
        }

        // Synchronize
        cudaDeviceSynchronize();

        // Count marked voxels
        int marked_voxels = 0;
        copyFromGPU();
        for (size_t i = 0; i < voxel_grid_.size(); ++i) {
            if (voxel_grid_[i] == cost_value) {
                marked_voxels++;
            }
        }

        return marked_voxels;
    }

    bool GPUCudaVoxelization::worldToVoxel(double world_x, double world_y, double world_z,
        int& voxel_x, int& voxel_y, int& voxel_z) const {
        voxel_x = static_cast<int>((world_x - origin_x_) / resolution_xy_);
        voxel_y = static_cast<int>((world_y - origin_y_) / resolution_xy_);
        voxel_z = static_cast<int>((world_z - origin_z_) / resolution_z_);

        return isValidVoxel(voxel_x, voxel_y, voxel_z);
    }

    void GPUCudaVoxelization::voxelToWorld(int voxel_x, int voxel_y, int voxel_z,
        double& world_x, double& world_y, double& world_z) const {
        world_x = origin_x_ + (voxel_x + 0.5) * resolution_xy_;
        world_y = origin_y_ + (voxel_y + 0.5) * resolution_xy_;
        world_z = origin_z_ + (voxel_z + 0.5) * resolution_z_;
    }

    bool GPUCudaVoxelization::isValidVoxel(int voxel_x, int voxel_y, int voxel_z) const {
        return (voxel_x >= 0 && voxel_x < grid_x_ &&
            voxel_y >= 0 && voxel_y < grid_y_ &&
            voxel_z >= 0 && voxel_z < grid_z_);
    }

    void GPUCudaVoxelization::updateVoxelCost(int voxel_x, int voxel_y, int voxel_z, unsigned char cost_value) {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            voxel_grid_[index] = cost_value;
        }
    }

    unsigned char GPUCudaVoxelization::getVoxelCost(int voxel_x, int voxel_y, int voxel_z) const {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            return voxel_grid_[index];
        }
        return 0;
    }

    void GPUCudaVoxelization::clear() {
        std::fill(voxel_grid_.begin(), voxel_grid_.end(), 0);
    }

    bool GPUCudaVoxelization::saveToFile(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Write grid dimensions
        file.write(reinterpret_cast<const char*>(&grid_x_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&grid_y_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&grid_z_), sizeof(int));

        // Write resolutions
        file.write(reinterpret_cast<const char*>(&resolution_xy_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&resolution_z_), sizeof(double));

        // Write origin
        file.write(reinterpret_cast<const char*>(&origin_x_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&origin_y_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&origin_z_), sizeof(double));

        // Write voxel grid
        file.write(reinterpret_cast<const char*>(voxel_grid_.data()), voxel_grid_.size());

        return true;
    }

    bool GPUCudaVoxelization::loadFromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Read grid dimensions
        file.read(reinterpret_cast<char*>(&grid_x_), sizeof(int));
        file.read(reinterpret_cast<char*>(&grid_y_), sizeof(int));
        file.read(reinterpret_cast<char*>(&grid_z_), sizeof(int));

        // Read resolutions
        file.read(reinterpret_cast<char*>(&resolution_xy_), sizeof(double));
        file.read(reinterpret_cast<char*>(&resolution_z_), sizeof(double));

        // Read origin
        file.read(reinterpret_cast<char*>(&origin_x_), sizeof(double));
        file.read(reinterpret_cast<char*>(&origin_y_), sizeof(double));
        file.read(reinterpret_cast<char*>(&origin_z_), sizeof(double));

        // Read voxel grid
        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_);
        file.read(reinterpret_cast<char*>(voxel_grid_.data()), voxel_grid_.size());

        return true;
    }

    // Legacy methods for compatibility
    int GPUCudaVoxelization::markBoxEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        return voxelizeBoxGPU(entity, buffer_size, cost_value);
    }

    int GPUCudaVoxelization::markCylinderEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        return voxelizeCylinderGPU(entity, buffer_size, cost_value);
    }

    int GPUCudaVoxelization::markSphereEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        return voxelizeSphereGPU(entity, buffer_size, cost_value);
    }

    void GPUCudaVoxelization::setResolution(int x, int y, int z) {
        grid_x_ = x; grid_y_ = y; grid_z_ = z;
        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_, 0);
        grid_size_ = voxel_grid_.size();
    }

    void GPUCudaVoxelization::setBoundingBox(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z) {
        origin_x_ = min_x;
        origin_y_ = min_y;
        origin_z_ = min_z;
        resolution_xy_ = (max_x - min_x) / grid_x_;
        resolution_z_ = (max_z - min_z) / grid_z_;
    }

    bool GPUCudaVoxelization::voxelize(const std::vector<std::shared_ptr<SpatialEntity>>& entities) {
        clear();
        return voxelizeEntities(entities) > 0;
    }

    bool GPUCudaVoxelization::saveVoxelGrid(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return false;
        }

        // Write grid dimensions
        file.write(reinterpret_cast<const char*>(&grid_x_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&grid_y_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&grid_z_), sizeof(int));

        // Write resolutions
        file.write(reinterpret_cast<const char*>(&resolution_xy_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&resolution_z_), sizeof(double));

        // Write origin
        file.write(reinterpret_cast<const char*>(&origin_x_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&origin_y_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&origin_z_), sizeof(double));

        // Write voxel grid data
        file.write(reinterpret_cast<const char*>(voxel_grid_.data()), voxel_grid_.size());

        file.close();
        std::cout << "Voxel grid saved to " << filename << std::endl;
        return true;
    }

} // namespace voxelization
