#include "gpu_voxelization.hpp"
#include "cpu_voxelization.hpp"
#include "spatial_entities.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

// TODO: This is a backup file for the GPU voxelization. It is not used in the project.
// It is kept here for reference.
// References for GPU-based voxelization:

// [1] Schwarz, Michael, and Hans-Peter Seidel. "Fast parallel surface and solid voxelization on GPUs."
//     ACM Transactions on Graphics (TOG) 29.6 (2010): 1-10.
//     DOI: https://doi.org/10.1145/1882261.1866201
//     - Presents efficient parallel algorithms for surface and solid voxelization on GPUs
//     - Introduces conservative and thin surface voxelization techniques
//     - Demonstrates high performance using CUDA

// [2] Pantaleoni, Jacopo. "VoxelPipe: a programmable pipeline for 3D voxelization."
//     Proceedings of the ACM SIGGRAPH Symposium on High Performance Graphics. 2011.
//     DOI: https://doi.org/10.1145/2018323.2018334
//     - Describes a flexible GPU pipeline for real-time voxelization
//     - Supports both conservative and thin voxelization
//     - Handles dynamic scenes efficiently

// [3] Crassin, Cyril, and Simon Green. "Octree-based sparse voxelization using the GPU hardware rasterizer."
//     OpenGL Insights (2012): 303-318.
//     - Details sparse voxelization techniques using GPU hardware rasterization
//     - Focuses on memory-efficient octree representations
//     - Provides practical implementation guidance for OpenGL

// [4] Zhang, Lei, et al. "Real-time voxelization for complex polygonal models."
//     Proceedings of the 12th Pacific Conference on Computer Graphics and Applications. IEEE, 2004.
//     DOI: https://doi.org/10.1109/PCCGA.2004.1348336
//     - Introduces slice-based parallel voxelization
//     - Demonstrates real-time performance for complex models
//     - Discusses optimization strategies

// [5] Karras, Tero, et al. "Real-time voxelization of complex scenes."
//     Proceedings of the 2010 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2010.
//     DOI: https://doi.org/10.1145/1833399.1833416
//     - Presents a real-time voxelization pipeline
//     - Demonstrates efficient handling of complex scenes
//     - Discusses performance optimizations

// [6] Karras, Tero, et al. "Real-time voxelization of complex scenes."
//     Proceedings of the 2010 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2010.
//     DOI: https://doi.org/10.1145/1833399.1833416
//     - Presents a real-time voxelization pipeline
//     - Demonstrates efficient handling of complex scenes
//     - Discusses performance optimizations
// CUDA includes with conditional compilation
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#endif

namespace voxelization {

#ifdef ENABLE_CUDA
    // CUDA kernel for voxelization
    __global__ void voxelizeBoxKernel(unsigned char* voxel_grid, int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z,
        double center_x, double center_y, double center_z,
        double size_x, double size_y, double size_z,
        unsigned char cost_value) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= grid_x || y >= grid_y || z >= grid_z) return;

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

    __global__ void voxelizeSphereKernel(unsigned char* voxel_grid, int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z,
        double center_x, double center_y, double center_z,
        double radius, unsigned char cost_value) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= grid_x || y >= grid_y || z >= grid_z) return;

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

    __global__ void voxelizeCylinderKernel(unsigned char* voxel_grid, int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z,
        double center_x, double center_y, double center_z,
        double radius, double height, unsigned char cost_value) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= grid_x || y >= grid_y || z >= grid_z) return;

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
#endif

    // GPUCudaVoxelization implementation
    GPUCudaVoxelization::GPUCudaVoxelization()
        : grid_x_(0), grid_y_(0), grid_z_(0), resolution_xy_(0.0), resolution_z_(0.0),
        origin_x_(0.0), origin_y_(0.0), origin_z_(0.0), d_voxel_grid_(nullptr),
        grid_size_(0), gpu_available_(false) {

#ifdef ENABLE_CUDA
        // Check CUDA availability
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error == cudaSuccess && deviceCount > 0) {
            gpu_available_ = true;
            std::cout << "CUDA GPU detected. GPU acceleration enabled." << std::endl;
        }
        else {
            gpu_available_ = false;
            std::cout << "CUDA GPU not available. Falling back to CPU implementation." << std::endl;
        }
#else
        gpu_available_ = false;
        std::cout << "CUDA support not compiled. Using CPU implementation." << std::endl;
#endif
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

    int GPUCudaVoxelization::voxelizeEntity(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {

        if (!gpu_available_) {
            // Fallback to CPU implementation
            CPUSequentialVoxelization cpu_voxelizer;
            cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);
            return cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
        }

#ifdef ENABLE_CUDA
        // GPU implementation
        clear();

        std::string entity_type = entity->getType();
        auto properties = entity->getProperties();

        if (entity_type == "box") {
            return voxelizeBoxGPU(entity, buffer_size, cost_value);
        }
        else if (entity_type == "sphere") {
            return voxelizeSphereGPU(entity, buffer_size, cost_value);
        }
        else if (entity_type == "cylinder") {
            return voxelizeCylinderGPU(entity, buffer_size, cost_value);
        }
        else {
            // For other entity types, use CPU implementation
            CPUSequentialVoxelization cpu_voxelizer;
            cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);
            return cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
        }
#else
        // Always use CPU implementation when CUDA is not available
        CPUSequentialVoxelization cpu_voxelizer;
        cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);
        return cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
#endif
    }

    int GPUCudaVoxelization::voxelizeEntities(const std::vector<std::shared_ptr<SpatialEntity>>& entities,
        double buffer_size, unsigned char cost_value) {

        if (!gpu_available_) {
            // Fallback to CPU implementation
            CPUSequentialVoxelization cpu_voxelizer;
            cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);
            return cpu_voxelizer.voxelizeEntities(entities, buffer_size, cost_value);
        }

#ifdef ENABLE_CUDA
        // GPU implementation
        clear();
        int total_voxels = 0;

        for (const auto& entity : entities) {
            total_voxels += voxelizeEntity(entity, buffer_size, cost_value);
        }

        return total_voxels;
#else
        // Always use CPU implementation when CUDA is not available
        CPUSequentialVoxelization cpu_voxelizer;
        cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);
        return cpu_voxelizer.voxelizeEntities(entities, buffer_size, cost_value);
#endif
    }

#ifdef ENABLE_CUDA
    int GPUCudaVoxelization::voxelizeBoxGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        auto properties = entity->getProperties();
        double center_x = properties["center_x"];
        double center_y = properties["center_y"];
        double center_z = properties["center_z"];
        double size_x = properties["size_x"];
        double size_y = properties["size_y"];
        double size_z = properties["size_z"];

        // Define block and grid dimensions
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);

        // Launch kernel
        voxelizeBoxKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z,
            size_x, size_y, size_z, cost_value);

        // Copy result back to host
        copyFromGPU();

        // Count marked voxels
        int marked_voxels = 0;
        for (size_t i = 0; i < voxel_grid_.size(); ++i) {
            if (voxel_grid_[i] == cost_value) {
                marked_voxels++;
            }
        }

        return marked_voxels;
    }

    int GPUCudaVoxelization::voxelizeSphereGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        auto properties = entity->getProperties();
        double center_x = properties["center_x"];
        double center_y = properties["center_y"];
        double center_z = properties["center_z"];
        double radius = properties["radius"];

        // Define block and grid dimensions
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);

        // Launch kernel
        voxelizeSphereKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z,
            radius, cost_value);

        // Copy result back to host
        copyFromGPU();

        // Count marked voxels
        int marked_voxels = 0;
        for (size_t i = 0; i < voxel_grid_.size(); ++i) {
            if (voxel_grid_[i] == cost_value) {
                marked_voxels++;
            }
        }

        return marked_voxels;
    }

    int GPUCudaVoxelization::voxelizeCylinderGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        auto properties = entity->getProperties();
        double center_x = properties["center_x"];
        double center_y = properties["center_y"];
        double center_z = properties["center_z"];
        double radius = properties["radius"];
        double height = properties["height"];

        // Define block and grid dimensions
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);

        // Launch kernel
        voxelizeCylinderKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z,
            radius, height, cost_value);

        // Copy result back to host
        copyFromGPU();

        // Count marked voxels
        int marked_voxels = 0;
        for (size_t i = 0; i < voxel_grid_.size(); ++i) {
            if (voxel_grid_[i] == cost_value) {
                marked_voxels++;
            }
        }

        return marked_voxels;
    }
#endif

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
#ifdef ENABLE_CUDA
        if (gpu_available_ && d_voxel_grid_) {
            cudaMemset(d_voxel_grid_, 0, grid_size_);
        }
#endif
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

    void GPUCudaVoxelization::allocateGPUMemory() {
#ifdef ENABLE_CUDA
        if (!gpu_available_) return;

        cudaError_t error = cudaMalloc(&d_voxel_grid_, grid_size_);
        if (error != cudaSuccess) {
            std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(error) << std::endl;
            gpu_available_ = false;
        }
#endif
    }

    void GPUCudaVoxelization::freeGPUMemory() {
#ifdef ENABLE_CUDA
        if (d_voxel_grid_) {
            cudaFree(d_voxel_grid_);
            d_voxel_grid_ = nullptr;
        }
#endif
    }

    void GPUCudaVoxelization::copyToGPU() {
#ifdef ENABLE_CUDA
        if (!gpu_available_ || !d_voxel_grid_) return;

        cudaError_t error = cudaMemcpy(d_voxel_grid_, voxel_grid_.data(), grid_size_, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cerr << "CUDA copy to device failed: " << cudaGetErrorString(error) << std::endl;
        }
#endif
    }

    void GPUCudaVoxelization::copyFromGPU() {
#ifdef ENABLE_CUDA
        if (!gpu_available_ || !d_voxel_grid_) return;

        cudaError_t error = cudaMemcpy(voxel_grid_.data(), d_voxel_grid_, grid_size_, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            std::cerr << "CUDA copy from device failed: " << cudaGetErrorString(error) << std::endl;
        }
#endif
    }

    // Legacy methods for compatibility
    int GPUCudaVoxelization::markBoxEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        return voxelizeBoxGPU(entity, buffer_size, cost_value);
#else
        return 0;
#endif
    }

    int GPUCudaVoxelization::markCylinderEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        return voxelizeCylinderGPU(entity, buffer_size, cost_value);
#else
        return 0;
#endif
    }

    int GPUCudaVoxelization::markSphereEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        return voxelizeSphereGPU(entity, buffer_size, cost_value);
#else
        return 0;
#endif
    }

    void GPUCudaVoxelization::setResolution(int x, int y, int z) {
        grid_x_ = x; grid_y_ = y; grid_z_ = z;
        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_, 0);
        grid_size_ = voxel_grid_.size();

        if (gpu_available_) {
            freeGPUMemory();
            allocateGPUMemory();
        }
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

} // namespace voxelization
