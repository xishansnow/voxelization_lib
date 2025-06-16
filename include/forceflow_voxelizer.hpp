#ifndef FORCEFLOW_VOXELIZER_HPP
#define FORCEFLOW_VOXELIZER_HPP

#include "voxelization_base.hpp"
#include "spatial_entities.hpp"
#include <vector>
#include <memory>
#include <cuda_runtime.h>

namespace voxelization {

    /**
     * @brief ForceFlow GPU voxelizer - specialized for high-performance mesh voxelization
     * Based on the efficient CUDA voxelizer implementation from ForceFlow
     * Optimized for polygon meshes with Morton encoding and bounding box optimization
     */
    class ForceFlowVoxelizer : public VoxelizationBase {
    public:
        ForceFlowVoxelizer();
        ~ForceFlowVoxelizer() override;

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

        // ForceFlow-specific methods
        bool isGPUAvailable() const { return gpu_available_; }
        bool isSolidVoxelization() const { return solid_voxelization_; }
        void setSolidVoxelization(bool solid) { solid_voxelization_ = solid; }

        void setResolution(int x, int y, int z) override;
        void setBoundingBox(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z) override;
        bool voxelize(const std::vector<std::shared_ptr<SpatialEntity>>& entities) override;
        bool saveVoxelGrid(const std::string& filename) const override;

        void getVoxelGrid(std::vector<unsigned char>& out) const;

    private:
        std::vector<unsigned char> voxel_grid_;
        int grid_x_, grid_y_, grid_z_;
        double resolution_xy_, resolution_z_;
        double origin_x_, origin_y_, origin_z_;

        // GPU memory pointers
        unsigned char* d_voxel_grid_;
        size_t grid_size_;

        bool gpu_available_;
        bool solid_voxelization_;  // Whether to use solid voxelization

        // ForceFlow-specific mesh voxelization
        int voxelizeMeshForceFlow(const std::shared_ptr<MeshEntity>& mesh, double buffer_size, unsigned char cost_value);

        // GPU memory management
        void allocateGPUMemory();
        void freeGPUMemory();
        void copyToGPU();
        void copyFromGPU();

        // Morton encoding utilities (for memory access optimization)
        unsigned int mortonEncode(unsigned int x, unsigned int y, unsigned int z) const;
        void mortonDecode(unsigned int morton, unsigned int& x, unsigned int& y, unsigned int& z) const;

        // Bounding box calculation for mesh
        std::vector<double> calculateMeshBoundingBox(const std::shared_ptr<MeshEntity>& mesh) const;

        // Fallback to CPU for non-mesh entities
        int fallbackToCPU(const std::shared_ptr<SpatialEntity>& entity, double buffer_size, unsigned char cost_value);
    };

} // namespace voxelization

#endif // FORCEFLOW_VOXELIZER_HPP
