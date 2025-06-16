#include "cpu_voxelization.hpp"
#include "spatial_entities.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace voxelization {

    // CPUSequentialVoxelization implementation
    CPUSequentialVoxelization::CPUSequentialVoxelization()
        : grid_x_(0), grid_y_(0), grid_z_(0), resolution_xy_(0.0), resolution_z_(0.0),
        origin_x_(0.0), origin_y_(0.0), origin_z_(0.0) {
    }

    CPUSequentialVoxelization::~CPUSequentialVoxelization() = default;

    void CPUSequentialVoxelization::initialize(int grid_x, int grid_y, int grid_z,
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
    }

    int CPUSequentialVoxelization::voxelizeEntity(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        if (!entity) return 0;

        std::string type = entity->getType();
        if (type == "box") {
            return markBoxEntity(entity, buffer_size, cost_value);
        }
        else if (type == "cylinder") {
            return markCylinderEntity(entity, buffer_size, cost_value);
        }
        else if (type == "sphere") {
            return markSphereEntity(entity, buffer_size, cost_value);
        }
        else if (type == "mesh") {
            return markMeshEntity(entity, buffer_size, cost_value);
        }

        // Generic approach for other entity types
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

#ifdef _OPENMP
#pragma omp parallel for reduction(+:marked_voxels) collapse(3)
#endif
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

    int CPUSequentialVoxelization::voxelizeEntities(const std::vector<std::shared_ptr<SpatialEntity>>& entities,
        double buffer_size, unsigned char cost_value) {
        int total_marked = 0;
        for (const auto& entity : entities) {
            total_marked += voxelizeEntity(entity, buffer_size, cost_value);
        }
        return total_marked;
    }

    bool CPUSequentialVoxelization::worldToVoxel(double world_x, double world_y, double world_z,
        int& voxel_x, int& voxel_y, int& voxel_z) const {
        voxel_x = static_cast<int>((world_x - origin_x_) / resolution_xy_);
        voxel_y = static_cast<int>((world_y - origin_y_) / resolution_xy_);
        voxel_z = static_cast<int>((world_z - origin_z_) / resolution_z_);

        return isValidVoxel(voxel_x, voxel_y, voxel_z);
    }

    void CPUSequentialVoxelization::voxelToWorld(int voxel_x, int voxel_y, int voxel_z,
        double& world_x, double& world_y, double& world_z) const {
        world_x = origin_x_ + (voxel_x + 0.5) * resolution_xy_;
        world_y = origin_y_ + (voxel_y + 0.5) * resolution_xy_;
        world_z = origin_z_ + (voxel_z + 0.5) * resolution_z_;
    }

    bool CPUSequentialVoxelization::isValidVoxel(int voxel_x, int voxel_y, int voxel_z) const {
        return (voxel_x >= 0 && voxel_x < grid_x_ &&
            voxel_y >= 0 && voxel_y < grid_y_ &&
            voxel_z >= 0 && voxel_z < grid_z_);
    }

    void CPUSequentialVoxelization::updateVoxelCost(int voxel_x, int voxel_y, int voxel_z, unsigned char cost_value) {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            voxel_grid_[index] = cost_value;
        }
    }

    unsigned char CPUSequentialVoxelization::getVoxelCost(int voxel_x, int voxel_y, int voxel_z) const {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            return voxel_grid_[index];
        }
        return 0;
    }

    void CPUSequentialVoxelization::clear() {
        std::fill(voxel_grid_.begin(), voxel_grid_.end(), 0);
    }

    bool CPUSequentialVoxelization::saveToFile(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Write header
        file.write(reinterpret_cast<const char*>(&grid_x_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&grid_y_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&grid_z_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&resolution_xy_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&resolution_z_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&origin_x_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&origin_y_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&origin_z_), sizeof(double));

        // Write voxel data
        file.write(reinterpret_cast<const char*>(voxel_grid_.data()), voxel_grid_.size());

        return true;
    }

    bool CPUSequentialVoxelization::loadFromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Read header
        file.read(reinterpret_cast<char*>(&grid_x_), sizeof(int));
        file.read(reinterpret_cast<char*>(&grid_y_), sizeof(int));
        file.read(reinterpret_cast<char*>(&grid_z_), sizeof(int));
        file.read(reinterpret_cast<char*>(&resolution_xy_), sizeof(double));
        file.read(reinterpret_cast<char*>(&resolution_z_), sizeof(double));
        file.read(reinterpret_cast<char*>(&origin_x_), sizeof(double));
        file.read(reinterpret_cast<char*>(&origin_y_), sizeof(double));
        file.read(reinterpret_cast<char*>(&origin_z_), sizeof(double));

        // Read voxel data
        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_);
        file.read(reinterpret_cast<char*>(voxel_grid_.data()), voxel_grid_.size());

        return true;
    }

    int CPUSequentialVoxelization::markBoxEntity(const std::shared_ptr<SpatialEntity>& entity,
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

        // Optimized serial implementation with better memory access pattern
        for (int z = min_z; z <= max_z; ++z) {
            // Pre-calculate z offset to avoid repeated multiplication
            int z_offset = z * grid_x_ * grid_y_;

            for (int y = min_y; y <= max_y; ++y) {
                // Pre-calculate y offset
                int y_offset = z_offset + y * grid_x_;

                for (int x = min_x; x <= max_x; ++x) {
                    double world_x, world_y, world_z;
                    voxelToWorld(x, y, z, world_x, world_y, world_z);

                    if (entity->isPointInside(world_x, world_y, world_z)) {
                        int index = y_offset + x;
                        voxel_grid_[index] = cost_value;
                        marked_voxels++;
                    }
                }
            }
        }

        return marked_voxels;
    }

    int CPUSequentialVoxelization::markCylinderEntity(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        return markBoxEntity(entity, buffer_size, cost_value);
    }

    int CPUSequentialVoxelization::markSphereEntity(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        return markBoxEntity(entity, buffer_size, cost_value);
    }

    int CPUSequentialVoxelization::markMeshEntity(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        return markBoxEntity(entity, buffer_size, cost_value);
    }

    void CPUSequentialVoxelization::setResolution(int x, int y, int z) {
        grid_x_ = x; grid_y_ = y; grid_z_ = z;
        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_, 0);
    }

    void CPUSequentialVoxelization::setBoundingBox(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z) {
        origin_x_ = min_x;
        origin_y_ = min_y;
        origin_z_ = min_z;
        resolution_xy_ = (max_x - min_x) / grid_x_;
        resolution_z_ = (max_z - min_z) / grid_z_;
    }

    bool CPUSequentialVoxelization::voxelize(const std::vector<std::shared_ptr<SpatialEntity>>& entities) {
        clear();
        return voxelizeEntities(entities) > 0;
    }

    bool CPUSequentialVoxelization::saveVoxelGrid(const std::string& filename) const {
        return saveToFile(filename);
    }

    // CPUParallelVoxelization implementation
    CPUParallelVoxelization::CPUParallelVoxelization()
        : grid_x_(0), grid_y_(0), grid_z_(0), resolution_xy_(0.0), resolution_z_(0.0),
        origin_x_(0.0), origin_y_(0.0), origin_z_(0.0) {

        // Set OpenMP thread count to number of CPU cores
#ifdef _OPENMP
        int num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
        std::cout << "CPU Parallel Voxelization initialized with " << num_threads << " threads" << std::endl;
#else
        std::cout << "CPU Parallel Voxelization initialized (OpenMP not available)" << std::endl;
#endif
    }

    CPUParallelVoxelization::~CPUParallelVoxelization() = default;

    void CPUParallelVoxelization::initialize(int grid_x, int grid_y, int grid_z,
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
    }

    int CPUParallelVoxelization::voxelizeEntity(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        if (!entity) return 0;

        std::string type = entity->getType();
        if (type == "box") {
            return markBoxEntityParallel(entity, buffer_size, cost_value);
        }
        else if (type == "cylinder") {
            return markCylinderEntityParallel(entity, buffer_size, cost_value);
        }
        else if (type == "sphere") {
            return markSphereEntityParallel(entity, buffer_size, cost_value);
        }
        else if (type == "mesh") {
            return markMeshEntityParallel(entity, buffer_size, cost_value);
        }

        // Generic approach for other entity types
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

#ifdef _OPENMP
#pragma omp parallel for reduction(+:marked_voxels) collapse(3)
#endif
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

    int CPUParallelVoxelization::voxelizeEntities(const std::vector<std::shared_ptr<SpatialEntity>>& entities,
        double buffer_size, unsigned char cost_value) {
        int total_marked = 0;
        for (const auto& entity : entities) {
            total_marked += voxelizeEntity(entity, buffer_size, cost_value);
        }
        return total_marked;
    }

    bool CPUParallelVoxelization::worldToVoxel(double world_x, double world_y, double world_z,
        int& voxel_x, int& voxel_y, int& voxel_z) const {
        voxel_x = static_cast<int>((world_x - origin_x_) / resolution_xy_);
        voxel_y = static_cast<int>((world_y - origin_y_) / resolution_xy_);
        voxel_z = static_cast<int>((world_z - origin_z_) / resolution_z_);

        return isValidVoxel(voxel_x, voxel_y, voxel_z);
    }

    void CPUParallelVoxelization::voxelToWorld(int voxel_x, int voxel_y, int voxel_z,
        double& world_x, double& world_y, double& world_z) const {
        world_x = origin_x_ + (voxel_x + 0.5) * resolution_xy_;
        world_y = origin_y_ + (voxel_y + 0.5) * resolution_xy_;
        world_z = origin_z_ + (voxel_z + 0.5) * resolution_z_;
    }

    bool CPUParallelVoxelization::isValidVoxel(int voxel_x, int voxel_y, int voxel_z) const {
        return (voxel_x >= 0 && voxel_x < grid_x_ &&
            voxel_y >= 0 && voxel_y < grid_y_ &&
            voxel_z >= 0 && voxel_z < grid_z_);
    }

    void CPUParallelVoxelization::updateVoxelCost(int voxel_x, int voxel_y, int voxel_z, unsigned char cost_value) {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            voxel_grid_[index] = cost_value;
        }
    }

    unsigned char CPUParallelVoxelization::getVoxelCost(int voxel_x, int voxel_y, int voxel_z) const {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            return voxel_grid_[index];
        }
        return 0;
    }

    void CPUParallelVoxelization::clear() {
        std::fill(voxel_grid_.begin(), voxel_grid_.end(), 0);
    }

    bool CPUParallelVoxelization::saveToFile(const std::string& filename) const {
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

    bool CPUParallelVoxelization::loadFromFile(const std::string& filename) {
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

    int CPUParallelVoxelization::markBoxEntityParallel(const std::shared_ptr<SpatialEntity>& entity,
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

        // Optimized parallel implementation with better memory access pattern
#ifdef _OPENMP
#pragma omp parallel for reduction(+:marked_voxels) schedule(dynamic, 1)
#endif
        for (int z = min_z; z <= max_z; ++z) {
            // Pre-calculate z offset to avoid repeated multiplication
            int z_offset = z * grid_x_ * grid_y_;

            for (int y = min_y; y <= max_y; ++y) {
                // Pre-calculate y offset
                int y_offset = z_offset + y * grid_x_;

                for (int x = min_x; x <= max_x; ++x) {
                    double world_x, world_y, world_z;
                    voxelToWorld(x, y, z, world_x, world_y, world_z);

                    if (entity->isPointInside(world_x, world_y, world_z)) {
                        int index = y_offset + x;
                        voxel_grid_[index] = cost_value;
                        marked_voxels++;
                    }
                }
            }
        }

        return marked_voxels;
    }

    int CPUParallelVoxelization::markCylinderEntityParallel(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        return markBoxEntityParallel(entity, buffer_size, cost_value);
    }

    int CPUParallelVoxelization::markSphereEntityParallel(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        return markBoxEntityParallel(entity, buffer_size, cost_value);
    }

    int CPUParallelVoxelization::markMeshEntityParallel(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        return markBoxEntityParallel(entity, buffer_size, cost_value);
    }

    void CPUParallelVoxelization::setResolution(int x, int y, int z) {
        grid_x_ = x; grid_y_ = y; grid_z_ = z;
        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_, 0);
    }

    void CPUParallelVoxelization::setBoundingBox(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z) {
        origin_x_ = min_x;
        origin_y_ = min_y;
        origin_z_ = min_z;
        resolution_xy_ = (max_x - min_x) / grid_x_;
        resolution_z_ = (max_z - min_z) / grid_z_;
    }

    bool CPUParallelVoxelization::voxelize(const std::vector<std::shared_ptr<SpatialEntity>>& entities) {
        clear();
        return voxelizeEntities(entities) > 0;
    }

    bool CPUParallelVoxelization::saveVoxelGrid(const std::string& filename) const {
        return saveToFile(filename);
    }

} // namespace voxelization
