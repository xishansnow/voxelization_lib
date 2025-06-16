#ifndef GPU_VOXELIZATION_HPP
#define GPU_VOXELIZATION_HPP

#include "voxelization_base.hpp"
#include <vector>
#include <memory>

namespace voxelization {

    /**
     * @brief GPU voxelization algorithm using CUDA
     */
    class GPUCudaVoxelization : public VoxelizationBase {
    public:
        GPUCudaVoxelization();
        ~GPUCudaVoxelization() override;

        void initialize(int grid_x, int grid_y, int grid_z,
            double resolution_xy, double resolution_z,
            double origin_x = 0.0, double origin_y = 0.0, double origin_z = 0.0) override;

        int voxelizeEntity(const std::shared_ptr<SpatialEntity>& entity,
            double buffer_size = 0.0,
            unsigned char cost_value = 255) override;

        int voxelizeEntities(const std::vector<std::shared_ptr<SpatialEntity>>& entities,
            double buffer_size = 0.0,
            unsigned char cost_value = 255) override;

        const unsigned char* getVoxelGrid() const override { return voxel_grid_.data(); }
        std::vector<int> getGridDimensions() const override { return { grid_x_, grid_y_, grid_z_ }; }
        std::vector<double> getResolutions() const override { return { resolution_xy_, resolution_z_ }; }

        bool worldToVoxel(double world_x, double world_y, double world_z,
            int& voxel_x, int& voxel_y, int& voxel_z) const override;

        void voxelToWorld(int voxel_x, int voxel_y, int voxel_z,
            double& world_x, double& world_y, double& world_z) const override;

        bool isValidVoxel(int voxel_x, int voxel_y, int voxel_z) const override;
        void updateVoxelCost(int voxel_x, int voxel_y, int voxel_z, unsigned char cost_value) override;
        unsigned char getVoxelCost(int voxel_x, int voxel_y, int voxel_z) const override;
        void clear() override;

        bool saveToFile(const std::string& filename) const override;
        bool loadFromFile(const std::string& filename) override;

        // GPU-specific methods
        bool isGPUAvailable() const { return gpu_available_; }

        // GPU voxelization methods for specific entity types
        int voxelizeBoxGPU(const std::shared_ptr<SpatialEntity>& entity,
            double buffer_size, unsigned char cost_value);
        int voxelizeSphereGPU(const std::shared_ptr<SpatialEntity>& entity,
            double buffer_size, unsigned char cost_value);
        int voxelizeCylinderGPU(const std::shared_ptr<SpatialEntity>& entity,
            double buffer_size, unsigned char cost_value);

        void setResolution(int x, int y, int z) override;
        void setBoundingBox(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z) override;
        bool voxelize(const std::vector<std::shared_ptr<SpatialEntity>>& entities) override;
        bool saveVoxelGrid(const std::string& filename) const override;

    private:
        std::vector<unsigned char> voxel_grid_;
        int grid_x_, grid_y_, grid_z_;
        double resolution_xy_, resolution_z_;
        double origin_x_, origin_y_, origin_z_;

        // GPU memory pointers
        unsigned char* d_voxel_grid_;
        size_t grid_size_;

        bool gpu_available_;

        void allocateGPUMemory();
        void freeGPUMemory();
        void copyToGPU();
        void copyFromGPU();

        int markBoxEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
            double buffer_size, unsigned char cost_value);
        int markCylinderEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
            double buffer_size, unsigned char cost_value);
        int markSphereEntityGPU(const std::shared_ptr<SpatialEntity>& entity,
            double buffer_size, unsigned char cost_value);
    };

} // namespace voxelization

#endif // GPU_VOXELIZATION_HPP
