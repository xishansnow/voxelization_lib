#include "gpu_voxelization.hpp"
#include "cpu_voxelization.hpp"
#include "spatial_entities.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

namespace voxelization {

    // Simplified GPU implementation using CPU threads to simulate GPU parallelism
    class SimulatedGPU {
    private:
        std::vector<unsigned char>& voxel_grid_;
        int grid_x_, grid_y_, grid_z_;
        double resolution_xy_, resolution_z_;
        double origin_x_, origin_y_, origin_z_;
        int num_threads_;

    public:
        SimulatedGPU(std::vector<unsigned char>& voxel_grid,
            int grid_x, int grid_y, int grid_z,
            double resolution_xy, double resolution_z,
            double origin_x, double origin_y, double origin_z)
            : voxel_grid_(voxel_grid), grid_x_(grid_x), grid_y_(grid_y), grid_z_(grid_z),
            resolution_xy_(resolution_xy), resolution_z_(resolution_z),
            origin_x_(origin_x), origin_y_(origin_y), origin_z_(origin_z) {
            num_threads_ = std::thread::hardware_concurrency();
        }

        // Simulate GPU kernel for box voxelization
        void voxelizeBox(double center_x, double center_y, double center_z,
            double size_x, double size_y, double size_z,
            unsigned char cost_value) {

            std::vector<std::thread> threads;
            int z_per_thread = grid_z_ / num_threads_;

            for (int t = 0; t < num_threads_; ++t) {
                int z_start = t * z_per_thread;
                int z_end = (t == num_threads_ - 1) ? grid_z_ : (t + 1) * z_per_thread;

                threads.emplace_back([=, this]() {
                    for (int z = z_start; z < z_end; ++z) {
                        for (int y = 0; y < grid_y_; ++y) {
                            for (int x = 0; x < grid_x_; ++x) {
                                // Convert voxel coordinates to world coordinates
                                double world_x = origin_x_ + (x + 0.5) * resolution_xy_;
                                double world_y = origin_y_ + (y + 0.5) * resolution_xy_;
                                double world_z = origin_z_ + (z + 0.5) * resolution_z_;

                                // Check if point is inside box
                                if (world_x >= center_x - size_x / 2 && world_x <= center_x + size_x / 2 &&
                                    world_y >= center_y - size_y / 2 && world_y <= center_y + size_y / 2 &&
                                    world_z >= center_z - size_z / 2 && world_z <= center_z + size_z / 2) {

                                    int index = z * grid_x_ * grid_y_ + y * grid_x_ + x;
                                    voxel_grid_[index] = cost_value;
                                }
                            }
                        }
                    }
                    });
            }

            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
        }

        // Simulate GPU kernel for sphere voxelization
        void voxelizeSphere(double center_x, double center_y, double center_z,
            double radius, unsigned char cost_value) {

            std::vector<std::thread> threads;
            int z_per_thread = grid_z_ / num_threads_;

            for (int t = 0; t < num_threads_; ++t) {
                int z_start = t * z_per_thread;
                int z_end = (t == num_threads_ - 1) ? grid_z_ : (t + 1) * z_per_thread;

                threads.emplace_back([=, this]() {
                    for (int z = z_start; z < z_end; ++z) {
                        for (int y = 0; y < grid_y_; ++y) {
                            for (int x = 0; x < grid_x_; ++x) {
                                // Convert voxel coordinates to world coordinates
                                double world_x = origin_x_ + (x + 0.5) * resolution_xy_;
                                double world_y = origin_y_ + (y + 0.5) * resolution_xy_;
                                double world_z = origin_z_ + (z + 0.5) * resolution_z_;

                                // Check if point is inside sphere
                                double dx = world_x - center_x;
                                double dy = world_y - center_y;
                                double dz = world_z - center_z;
                                double distance_squared = dx * dx + dy * dy + dz * dz;

                                if (distance_squared <= radius * radius) {
                                    int index = z * grid_x_ * grid_y_ + y * grid_x_ + x;
                                    voxel_grid_[index] = cost_value;
                                }
                            }
                        }
                    }
                    });
            }

            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
        }

        // Simulate GPU kernel for cylinder voxelization
        void voxelizeCylinder(double center_x, double center_y, double center_z,
            double radius, double height, unsigned char cost_value) {

            std::vector<std::thread> threads;
            int z_per_thread = grid_z_ / num_threads_;

            for (int t = 0; t < num_threads_; ++t) {
                int z_start = t * z_per_thread;
                int z_end = (t == num_threads_ - 1) ? grid_z_ : (t + 1) * z_per_thread;

                threads.emplace_back([=, this]() {
                    for (int z = z_start; z < z_end; ++z) {
                        for (int y = 0; y < grid_y_; ++y) {
                            for (int x = 0; x < grid_x_; ++x) {
                                // Convert voxel coordinates to world coordinates
                                double world_x = origin_x_ + (x + 0.5) * resolution_xy_;
                                double world_y = origin_y_ + (y + 0.5) * resolution_xy_;
                                double world_z = origin_z_ + (z + 0.5) * resolution_z_;

                                // Check if point is inside cylinder
                                double dx = world_x - center_x;
                                double dy = world_y - center_y;
                                double distance_squared = dx * dx + dy * dy;

                                if (distance_squared <= radius * radius &&
                                    world_z >= center_z - height / 2 && world_z <= center_z + height / 2) {
                                    int index = z * grid_x_ * grid_y_ + y * grid_x_ + x;
                                    voxel_grid_[index] = cost_value;
                                }
                            }
                        }
                    }
                    });
            }

            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
        }
    };

    // GPUCudaVoxelization implementation
    GPUCudaVoxelization::GPUCudaVoxelization()
        : grid_x_(0), grid_y_(0), grid_z_(0), resolution_xy_(0.0), resolution_z_(0.0),
        origin_x_(0.0), origin_y_(0.0), origin_z_(0.0), d_voxel_grid_(nullptr),
        grid_size_(0), gpu_available_(false) {

        // Check if we can use simulated GPU (multi-threaded CPU)
        gpu_available_ = true;
        std::cout << "Simulated GPU acceleration enabled using "
            << std::thread::hardware_concurrency() << " threads." << std::endl;
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

        // Simulated GPU implementation
        clear();

        std::string entity_type = entity->getType();
        auto properties = entity->getProperties();

        SimulatedGPU gpu(voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);

        if (entity_type == "box") {
            double center_x = properties["center_x"];
            double center_y = properties["center_y"];
            double center_z = properties["center_z"];
            double size_x = properties["size_x"];
            double size_y = properties["size_y"];
            double size_z = properties["size_z"];

            gpu.voxelizeBox(center_x, center_y, center_z, size_x, size_y, size_z, cost_value);

        }
        else if (entity_type == "sphere") {
            double center_x = properties["center_x"];
            double center_y = properties["center_y"];
            double center_z = properties["center_z"];
            double radius = properties["radius"];

            gpu.voxelizeSphere(center_x, center_y, center_z, radius, cost_value);

        }
        else if (entity_type == "cylinder") {
            double center_x = properties["center_x"];
            double center_y = properties["center_y"];
            double center_z = properties["center_z"];
            double radius = properties["radius"];
            double height = properties["height"];

            gpu.voxelizeCylinder(center_x, center_y, center_z, radius, height, cost_value);

        }
        else {
            // For other entity types, use CPU implementation
            CPUSequentialVoxelization cpu_voxelizer;
            cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);
            return cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
        }

        // Count marked voxels
        int marked_voxels = 0;
        for (size_t i = 0; i < voxel_grid_.size(); ++i) {
            if (voxel_grid_[i] == cost_value) {
                marked_voxels++;
            }
        }

        return marked_voxels;
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

        // Simulated GPU implementation
        clear();
        int total_voxels = 0;

        for (const auto& entity : entities) {
            total_voxels += voxelizeEntity(entity, buffer_size, cost_value);
        }

        return total_voxels;
    }

    int GPUCudaVoxelization::voxelizeBoxGPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {
        auto properties = entity->getProperties();
        double center_x = properties["center_x"];
        double center_y = properties["center_y"];
        double center_z = properties["center_z"];
        double size_x = properties["size_x"];
        double size_y = properties["size_y"];
        double size_z = properties["size_z"];

        SimulatedGPU gpu(voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);

        gpu.voxelizeBox(center_x, center_y, center_z, size_x, size_y, size_z, cost_value);

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

        SimulatedGPU gpu(voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);

        gpu.voxelizeSphere(center_x, center_y, center_z, radius, cost_value);

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

        SimulatedGPU gpu(voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);

        gpu.voxelizeCylinder(center_x, center_y, center_z, radius, height, cost_value);

        // Count marked voxels
        int marked_voxels = 0;
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

    void GPUCudaVoxelization::allocateGPUMemory() {
        // No-op for simulated GPU
    }

    void GPUCudaVoxelization::freeGPUMemory() {
        // No-op for simulated GPU
    }

    void GPUCudaVoxelization::copyToGPU() {
        // No-op for simulated GPU
    }

    void GPUCudaVoxelization::copyFromGPU() {
        // No-op for simulated GPU
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

} // namespace voxelization
