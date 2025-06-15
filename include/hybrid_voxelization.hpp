#ifndef HYBRID_VOXELIZATION_HPP
#define HYBRID_VOXELIZATION_HPP

#include "voxelization_base.hpp"
#include "cpu_voxelization.hpp"
#include "gpu_voxelization.hpp"
#include <vector>
#include <memory>

namespace voxelization {

    /**
     * @brief Hybrid voxelization algorithm (CPU + GPU)
     */
    class HybridVoxelization : public VoxelizationBase {
    public:
        // Strategy selection
        enum class Strategy {
            CPU_ONLY,
            GPU_ONLY,
            HYBRID
        };

        HybridVoxelization();
        ~HybridVoxelization() override;

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

        // Hybrid-specific methods
        bool isGPUAvailable() const { return gpu_available_; }
        void setStrategy(Strategy strategy) { current_strategy_ = strategy; }
        Strategy getCurrentStrategy() const { return current_strategy_; }

        void setResolution(int x, int y, int z) override;
        void setBoundingBox(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z) override;
        bool voxelize(const std::vector<std::shared_ptr<SpatialEntity>>& entities) override;
        bool saveVoxelGrid(const std::string& filename) const override;

    private:
        std::vector<unsigned char> voxel_grid_;
        int grid_x_, grid_y_, grid_z_;
        double resolution_xy_, resolution_z_;
        double origin_x_, origin_y_, origin_z_;

        std::unique_ptr<CPUSequentialVoxelization> cpu_voxelizer_;
        std::unique_ptr<GPUCudaVoxelization> gpu_voxelizer_;

        bool gpu_available_;
        Strategy current_strategy_;

        Strategy selectStrategy(const std::shared_ptr<SpatialEntity>& entity) const;
        Strategy selectStrategy(const std::vector<std::shared_ptr<SpatialEntity>>& entities) const;
        void syncVoxelGrid();
    };

} // namespace voxelization

#endif // HYBRID_VOXELIZATION_HPP
