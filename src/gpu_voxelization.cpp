#include "gpu_voxelization.hpp"
#include "cpu_voxelization.hpp"
#include "spatial_entities.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

namespace voxelization {

    // GPUCudaVoxelization implementation (stub for now)
    GPUCudaVoxelization::GPUCudaVoxelization()
        : grid_x_(0), grid_y_(0), grid_z_(0), resolution_xy_(0.0), resolution_z_(0.0),
        origin_x_(0.0), origin_y_(0.0), origin_z_(0.0), d_voxel_grid_(nullptr),
        grid_size_(0), gpu_available_(false) {
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

        // For now, just use CPU implementation
        gpu_available_ = false;
    }

    int GPUCudaVoxelization::voxelizeEntity(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        // For now, use CPU implementation
        CPUSequentialVoxelization cpu_voxelizer;
        cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);
        return cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
    }

    int GPUCudaVoxelization::voxelizeEntities(const std::vector<std::shared_ptr<SpatialEntity>>& entities,
        double buffer_size, unsigned char cost_value) {
        // For now, use CPU implementation
        CPUSequentialVoxelization cpu_voxelizer;
        cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);
        return cpu_voxelizer.voxelizeEntities(entities, buffer_size, cost_value);
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

    void GPUCudaVoxelization::allocateGPUMemory() {
        // CUDA implementation would go here
        // cudaMalloc(&d_voxel_grid_, grid_size_);
    }

    void GPUCudaVoxelization::freeGPUMemory() {
        // CUDA implementation would go here
        // if (d_voxel_grid_) {
        //     cudaFree(d_voxel_grid_);
        //     d_voxel_grid_ = nullptr;
        // }
    }

    void GPUCudaVoxelization::copyToGPU() {
        // CUDA implementation would go here
        // cudaMemcpy(d_voxel_grid_, voxel_grid_.data(), grid_size_, cudaMemcpyHostToDevice);
    }

    void GPUCudaVoxelization::copyFromGPU() {
        // CUDA implementation would go here
        // cudaMemcpy(voxel_grid_.data(), d_voxel_grid_, grid_size_, cudaMemcpyDeviceToHost);
    }

    int GPUCudaVoxelization::markBoxEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        auto bbox = entity->getBoundingBox();
        int min_x = static_cast<int>((bbox[0] - buffer_size - origin_x_) / resolution_xy_);
        int max_x = static_cast<int>((bbox[3] + buffer_size - origin_x_) / resolution_xy_);
        int min_y = static_cast<int>((bbox[1] - buffer_size - origin_y_) / resolution_xy_);
        int max_y = static_cast<int>((bbox[4] + buffer_size - origin_y_) / resolution_xy_);
        int min_z = static_cast<int>((bbox[2] - buffer_size - origin_z_) / resolution_z_);
        int max_z = static_cast<int>((bbox[5] + buffer_size - origin_z_) / resolution_z_);

        // Clamp to grid bounds
        min_x = std::max(0, std::min(grid_x_ - 1, min_x));
        max_x = std::max(0, std::min(grid_x_ - 1, max_x));
        min_y = std::max(0, std::min(grid_y_ - 1, min_y));
        max_y = std::max(0, std::min(grid_y_ - 1, max_y));
        min_z = std::max(0, std::min(grid_z_ - 1, min_z));
        max_z = std::max(0, std::min(grid_z_ - 1, max_z));

        int marked_voxels = 0;

        for (int z = min_z; z <= max_z; ++z) {
            for (int y = min_y; y <= max_y; ++y) {
                for (int x = min_x; x <= max_x; ++x) {
                    double world_x, world_y, world_z;
                    voxelToWorld(x, y, z, world_x, world_y, world_z);

                    if (entity->isPointInside(world_x, world_y, world_z)) {
                        int index = z * grid_x_ * grid_y_ + y * grid_x_ + x;
                        voxel_grid_[index] = cost_value;
                        marked_voxels++;
                    }
                }
            }
        }

        return marked_voxels;
    }

    int GPUCudaVoxelization::markCylinderEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        return markBoxEntityGPU(entity, buffer_size, cost_value);
    }

    int GPUCudaVoxelization::markSphereEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        return markBoxEntityGPU(entity, buffer_size, cost_value);
    }

    void GPUCudaVoxelization::setResolution(int x, int y, int z) {
        grid_x_ = x; grid_y_ = y; grid_z_ = z;
        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_, 0);
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
        return saveToFile(filename);
    }

} // namespace voxelization
