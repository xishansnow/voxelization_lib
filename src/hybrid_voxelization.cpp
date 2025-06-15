#include "hybrid_voxelization.hpp"
#include "spatial_entities.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

namespace voxelization {

    // HybridVoxelization implementation (stub for now)
    HybridVoxelization::HybridVoxelization()
        : grid_x_(0), grid_y_(0), grid_z_(0), resolution_xy_(0.0), resolution_z_(0.0),
        origin_x_(0.0), origin_y_(0.0), origin_z_(0.0), gpu_available_(false),
        current_strategy_(Strategy::CPU_ONLY) {
    }

    HybridVoxelization::~HybridVoxelization() = default;

    void HybridVoxelization::initialize(int grid_x, int grid_y, int grid_z,
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

        // Initialize CPU and GPU voxelizers
        cpu_voxelizer_ = std::make_unique<CPUSequentialVoxelization>();
        gpu_voxelizer_ = std::make_unique<GPUCudaVoxelization>();

        cpu_voxelizer_->initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);
        gpu_voxelizer_->initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);

        gpu_available_ = false; // For now
        current_strategy_ = Strategy::CPU_ONLY;
    }

    int HybridVoxelization::voxelizeEntity(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        Strategy strategy = selectStrategy(entity);

        switch (strategy) {
        case Strategy::CPU_ONLY:
            return cpu_voxelizer_->voxelizeEntity(entity, buffer_size, cost_value);

        case Strategy::GPU_ONLY:
            return gpu_voxelizer_->voxelizeEntity(entity, buffer_size, cost_value);

        case Strategy::HYBRID:
            // For now, use CPU
            return cpu_voxelizer_->voxelizeEntity(entity, buffer_size, cost_value);

        default:
            return cpu_voxelizer_->voxelizeEntity(entity, buffer_size, cost_value);
        }
    }

    int HybridVoxelization::voxelizeEntities(const std::vector<std::shared_ptr<SpatialEntity>>& entities,
        double buffer_size, unsigned char cost_value) {
        Strategy strategy = selectStrategy(entities);

        switch (strategy) {
        case Strategy::CPU_ONLY:
            return cpu_voxelizer_->voxelizeEntities(entities, buffer_size, cost_value);

        case Strategy::GPU_ONLY:
            return gpu_voxelizer_->voxelizeEntities(entities, buffer_size, cost_value);

        case Strategy::HYBRID:
            // For now, use CPU
            return cpu_voxelizer_->voxelizeEntities(entities, buffer_size, cost_value);

        default:
            return cpu_voxelizer_->voxelizeEntities(entities, buffer_size, cost_value);
        }
    }

    bool HybridVoxelization::worldToVoxel(double world_x, double world_y, double world_z,
        int& voxel_x, int& voxel_y, int& voxel_z) const {
        voxel_x = static_cast<int>((world_x - origin_x_) / resolution_xy_);
        voxel_y = static_cast<int>((world_y - origin_y_) / resolution_xy_);
        voxel_z = static_cast<int>((world_z - origin_z_) / resolution_z_);

        return isValidVoxel(voxel_x, voxel_y, voxel_z);
    }

    void HybridVoxelization::voxelToWorld(int voxel_x, int voxel_y, int voxel_z,
        double& world_x, double& world_y, double& world_z) const {
        world_x = origin_x_ + (voxel_x + 0.5) * resolution_xy_;
        world_y = origin_y_ + (voxel_y + 0.5) * resolution_xy_;
        world_z = origin_z_ + (voxel_z + 0.5) * resolution_z_;
    }

    bool HybridVoxelization::isValidVoxel(int voxel_x, int voxel_y, int voxel_z) const {
        return (voxel_x >= 0 && voxel_x < grid_x_ &&
            voxel_y >= 0 && voxel_y < grid_y_ &&
            voxel_z >= 0 && voxel_z < grid_z_);
    }

    void HybridVoxelization::updateVoxelCost(int voxel_x, int voxel_y, int voxel_z, unsigned char cost_value) {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            voxel_grid_[index] = cost_value;
        }
    }

    unsigned char HybridVoxelization::getVoxelCost(int voxel_x, int voxel_y, int voxel_z) const {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            return voxel_grid_[index];
        }
        return 0;
    }

    void HybridVoxelization::clear() {
        std::fill(voxel_grid_.begin(), voxel_grid_.end(), 0);
        if (cpu_voxelizer_) cpu_voxelizer_->clear();
        if (gpu_voxelizer_) gpu_voxelizer_->clear();
    }

    bool HybridVoxelization::saveToFile(const std::string& filename) const {
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

    bool HybridVoxelization::loadFromFile(const std::string& filename) {
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

    HybridVoxelization::Strategy HybridVoxelization::selectStrategy(const std::shared_ptr<SpatialEntity>& entity) const {
        // Simple strategy: use GPU for large entities, CPU for small ones
        auto bbox = entity->getBoundingBox();
        double volume = (bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (bbox[5] - bbox[2]);

        if (volume > 1000.0 && gpu_available_) {
            return Strategy::GPU_ONLY;
        }
        else {
            return Strategy::CPU_ONLY;
        }
    }

    HybridVoxelization::Strategy HybridVoxelization::selectStrategy(const std::vector<std::shared_ptr<SpatialEntity>>& entities) const {
        // Simple strategy: use GPU for many entities, CPU for few
        if (entities.size() > 10 && gpu_available_) {
            return Strategy::GPU_ONLY;
        }
        else {
            return Strategy::CPU_ONLY;
        }
    }

    void HybridVoxelization::syncVoxelGrid() {
        // Synchronize voxel grid between CPU and GPU implementations
        // This would be implemented when actual GPU support is added
    }

    void HybridVoxelization::setResolution(int x, int y, int z) {
        grid_x_ = x; grid_y_ = y; grid_z_ = z;
        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_, 0);
    }

    void HybridVoxelization::setBoundingBox(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z) {
        origin_x_ = min_x;
        origin_y_ = min_y;
        origin_z_ = min_z;
        resolution_xy_ = (max_x - min_x) / grid_x_;
        resolution_z_ = (max_z - min_z) / grid_z_;
    }

    bool HybridVoxelization::voxelize(const std::vector<std::shared_ptr<SpatialEntity>>& entities) {
        clear();
        return voxelizeEntities(entities) > 0;
    }

    bool HybridVoxelization::saveVoxelGrid(const std::string& filename) const {
        return saveToFile(filename);
    }

} // namespace voxelization
