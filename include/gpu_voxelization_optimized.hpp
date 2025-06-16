#ifndef GPU_VOXELIZATION_OPTIMIZED_HPP
#define GPU_VOXELIZATION_OPTIMIZED_HPP

#include "voxelization_base.hpp"
#include "spatial_entities.hpp"
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace voxelization {

    // 前向声明
    class OptimizedGPUVoxelization;

    /**
     * @brief 高度优化的GPU体素化类
     *
     * 实现了多种优化策略：
     * 1. 空间划分 - 使用八叉树进行空间划分
     * 2. Morton编码 - 提高内存访问局部性
     * 3. 2-Pass策略 - 粗粒度+细粒度体素化
     * 4. 共享内存优化 - 减少全局内存访问
     * 5. 局部缓冲区 - 提高缓存命中率
     */
    class OptimizedGPUCudaVoxelization : public VoxelizationBase {
    public:
        OptimizedGPUCudaVoxelization();
        ~OptimizedGPUCudaVoxelization();

        // 基础接口实现
        void initialize(int grid_x, int grid_y, int grid_z,
            double resolution_xy, double resolution_z,
            double origin_x = 0.0, double origin_y = 0.0, double origin_z = 0.0) override;

        void addEntity(const std::shared_ptr<SpatialEntity>& entity);
        void voxelize(unsigned char cost_value = 255);
        const unsigned char* getVoxelGrid() const;
        bool saveVoxelGrid(const std::string& filename) const;

        // 优化配置接口
        void setOptimizationFlags(bool use_octree = true, bool use_morton = true, bool use_2pass = true);
        void setCoarseFactor(int factor);
        void setBlockSize(int block_size_1d, int block_size_2d, int block_size_3d);
        void setSharedMemorySize(int size_kb);
        void setOctreeDepth(int depth);
        void setMortonBits(int bits);

        // 性能监控接口
        struct PerformanceMetrics {
            double octree_build_time;
            double morton_generation_time;
            double coarse_pass_time;
            double fine_pass_time;
            double total_time;
            size_t memory_usage;
            int active_voxels;
            int total_voxels;
        };

        PerformanceMetrics getPerformanceMetrics() const;
        void resetPerformanceMetrics();

        // 高级优化接口
        void enableAdaptiveResolution(bool enable);
        void setLoadBalancingStrategy(const std::string& strategy);
        void enableMemoryCoalescing(bool enable);
        void setWarpSize(int warp_size);

        // 继承自VoxelizationBase的所有纯虚函数
        int voxelizeEntity(const std::shared_ptr<SpatialEntity>& entity, double buffer_size, unsigned char cost_value) override;
        int voxelizeEntities(const std::vector<std::shared_ptr<SpatialEntity>>& entities, double buffer_size, unsigned char cost_value) override;
        std::vector<int> getGridDimensions() const override;
        std::vector<double> getResolutions() const override;
        bool worldToVoxel(double world_x, double world_y, double world_z, int& voxel_x, int& voxel_y, int& voxel_z) const override;
        void voxelToWorld(int voxel_x, int voxel_y, int voxel_z, double& world_x, double& world_y, double& world_z) const override;
        bool isValidVoxel(int voxel_x, int voxel_y, int voxel_z) const override;
        void updateVoxelCost(int voxel_x, int voxel_y, int voxel_z, unsigned char cost_value) override;
        unsigned char getVoxelCost(int voxel_x, int voxel_y, int voxel_z) const override;
        void clear() override;
        bool saveToFile(const std::string& filename) const override;
        bool loadFromFile(const std::string& filename) override;
        void setResolution(int x, int y, int z) override;
        void setBoundingBox(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z) override;
        bool voxelize(const std::vector<std::shared_ptr<SpatialEntity>>& entities) override;

        void getVoxelGrid(std::vector<unsigned char>& out) const;

    private:
        std::unique_ptr<OptimizedGPUVoxelization> impl_;
        PerformanceMetrics metrics_;

        // 配置参数
        bool use_octree_;
        bool use_morton_;
        bool use_2pass_;
        int coarse_factor_;
        int block_size_1d_;
        int block_size_2d_;
        int block_size_3d_;
        int shared_memory_size_kb_;
        int octree_depth_;
        int morton_bits_;
        bool adaptive_resolution_;
        std::string load_balancing_strategy_;
        bool memory_coalescing_;
        int warp_size_;
    };

    /**
     * @brief GPU内存管理优化器
     */
    class GPUMemoryOptimizer {
    public:
        GPUMemoryOptimizer();
        ~GPUMemoryOptimizer();

        // 内存池管理
        void* allocatePooledMemory(size_t size, size_t alignment = 256);
        void freePooledMemory(void* ptr);
        void clearMemoryPool();

        // 内存优化策略
        void enableUnifiedMemory(bool enable);
        void setMemoryPoolSize(size_t size_mb);
        void enableMemoryCompression(bool enable);
        void setMemoryAlignment(size_t alignment);

        // 内存统计
        struct MemoryStats {
            size_t total_allocated;
            size_t total_freed;
            size_t peak_usage;
            size_t current_usage;
            size_t pool_size;
            int allocation_count;
        };

        MemoryStats getMemoryStats() const;

    private:
        struct MemoryBlock {
            void* ptr;
            size_t size;
            bool in_use;
        };

        std::vector<MemoryBlock> memory_pool_;
        size_t pool_size_;
        size_t alignment_;
        bool unified_memory_;
        bool compression_enabled_;
        MemoryStats stats_;
    };

    /**
     * @brief GPU内核优化器
     */
    class GPUKernelOptimizer {
    public:
        GPUKernelOptimizer();
        ~GPUKernelOptimizer();

        // 内核配置优化
        struct KernelConfig {
            dim3 block_size;
            dim3 grid_size;
            size_t shared_memory_size;
            int max_blocks_per_sm;
            int warp_size;
        };

        KernelConfig optimizeKernelConfig(int grid_x, int grid_y, int grid_z,
            size_t shared_memory_required = 0);

        // 负载均衡
        void setLoadBalancingStrategy(const std::string& strategy);
        void enableDynamicParallelism(bool enable);
        void setMaxBlocksPerSM(int max_blocks);

        // 内核性能分析
        struct KernelProfile {
            double kernel_time;
            double memory_transfer_time;
            double occupancy;
            int active_warps;
            int total_warps;
            size_t shared_memory_usage;
            size_t register_usage;
        };

        KernelProfile profileKernel(const std::string& kernel_name);

    private:
        std::string load_balancing_strategy_;
        bool dynamic_parallelism_;
        int max_blocks_per_sm_;
        int warp_size_;
    };

    /**
     * @brief 空间数据结构优化器
     */
    class SpatialDataOptimizer {
    public:
        SpatialDataOptimizer();
        ~SpatialDataOptimizer();

        // 八叉树优化
        struct OctreeConfig {
            int max_depth;
            int max_entities_per_node;
            float split_threshold;
            bool adaptive_depth;
        };

        void setOctreeConfig(const OctreeConfig& config);
        OctreeConfig getOctreeConfig() const;

        // Morton编码优化
        struct MortonConfig {
            int bits_per_axis;
            bool interleaved;
            bool compressed;
        };

        void setMortonConfig(const MortonConfig& config);
        MortonConfig getMortonConfig() const;

        // 空间哈希优化
        struct SpatialHashConfig {
            float cell_size;
            int hash_table_size;
            bool linear_probing;
        };

        void setSpatialHashConfig(const SpatialHashConfig& config);
        SpatialHashConfig getSpatialHashConfig() const;

    private:
        OctreeConfig octree_config_;
        MortonConfig morton_config_;
        SpatialHashConfig spatial_hash_config_;
    };

} // namespace voxelization

#endif // GPU_VOXELIZATION_OPTIMIZED_HPP
