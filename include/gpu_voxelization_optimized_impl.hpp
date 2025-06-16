#pragma once
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "spatial_entities.hpp"

namespace voxelization {

    struct OptimizedEntity {
        int type; // 0: box, 1: sphere, 2: cylinder
        float3 center;
        float3 size; // 用于box
        float radius; // 用于sphere和cylinder
        float height; // 用于cylinder
    };

    class OptimizedGPUVoxelization {
    private:
        // GPU内存
        unsigned char* d_voxel_grid_;
        OptimizedEntity* d_entities_;
        // 主机内存
        std::vector<OptimizedEntity> h_entities_;
        mutable std::vector<unsigned char> h_voxel_grid_;
        // 网格参数
        int grid_x_, grid_y_, grid_z_;
        float resolution_;
        float3 origin_;
        size_t grid_size_;
    public:
        OptimizedGPUVoxelization();
        ~OptimizedGPUVoxelization();
        void initializeGrid(int grid_x, int grid_y, int grid_z, float resolution, float3 origin);
        void addEntity(const std::shared_ptr<SpatialEntity>& entity);
        void prepareOptimizations();
        void voxelize(unsigned char cost_value = 255);
        void getVoxelGrid(std::vector<unsigned char>& voxel_grid);
        const unsigned char* getVoxelGridPtr() const;
        bool saveVoxelGrid(const std::string& filename) const;
        void freeGPUMemory();
        void clear();
        float3 getOrigin() const { return origin_; }
        float getResolution() const { return resolution_; }
        int getGridX() const { return grid_x_; }
        int getGridY() const { return grid_y_; }
        int getGridZ() const { return grid_z_; }
    };

} // namespace voxelization
