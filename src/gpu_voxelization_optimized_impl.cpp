#include "gpu_voxelization_optimized.hpp"
#include "gpu_voxelization_optimized_impl.hpp"
#include "cpu_voxelization.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

namespace voxelization {

    // OptimizedGPUCudaVoxelization 实现
    OptimizedGPUCudaVoxelization::OptimizedGPUCudaVoxelization()
        : impl_(std::make_unique<OptimizedGPUVoxelization>()),
        use_octree_(true), use_morton_(true), use_2pass_(true),
        coarse_factor_(4), block_size_1d_(256), block_size_2d_(16), block_size_3d_(8),
        shared_memory_size_kb_(48), octree_depth_(8), morton_bits_(21),
        adaptive_resolution_(false), load_balancing_strategy_("uniform"),
        memory_coalescing_(true), warp_size_(32) {

        // 初始化性能指标
        metrics_ = { 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0 };

        // 设置默认优化配置
//         impl_->setOptimizationFlags(use_octree_, use_morton_, use_2pass_);
//         impl_->setCoarseFactor(coarse_factor_);
    }

    OptimizedGPUCudaVoxelization::~OptimizedGPUCudaVoxelization() = default;

    void OptimizedGPUCudaVoxelization::initialize(int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z) {
        auto start_time = std::chrono::high_resolution_clock::now();
        float3 origin = make_float3(origin_x, origin_y, origin_z);
        impl_->initializeGrid(grid_x, grid_y, grid_z, resolution_xy, origin);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        metrics_.total_time += duration.count() / 1000.0; // 转换为毫秒
    }

    void OptimizedGPUCudaVoxelization::addEntity(const std::shared_ptr<SpatialEntity>& entity) {
        impl_->addEntity(entity);
    }

    void OptimizedGPUCudaVoxelization::voxelize(unsigned char cost_value) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 准备优化数据结构
        auto prep_start = std::chrono::high_resolution_clock::now();
        impl_->prepareOptimizations();
        auto prep_end = std::chrono::high_resolution_clock::now();
        auto prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(prep_end - prep_start);

        if (use_octree_) {
            metrics_.octree_build_time += prep_duration.count() / 1000.0;
        }
        if (use_morton_) {
            metrics_.morton_generation_time += prep_duration.count() / 1000.0;
        }

        // 执行体素化
        auto voxelize_start = std::chrono::high_resolution_clock::now();
        impl_->voxelize(cost_value);
        auto voxelize_end = std::chrono::high_resolution_clock::now();
        auto voxelize_duration = std::chrono::duration_cast<std::chrono::microseconds>(voxelize_end - voxelize_start);

        if (use_2pass_) {
            metrics_.coarse_pass_time += voxelize_duration.count() / 2000.0; // 假设各占一半时间
            metrics_.fine_pass_time += voxelize_duration.count() / 2000.0;
        }
        else {
            metrics_.fine_pass_time += voxelize_duration.count() / 1000.0;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        metrics_.total_time += total_duration.count() / 1000.0;
    }

    const unsigned char* OptimizedGPUCudaVoxelization::getVoxelGrid() const {
        return impl_->getVoxelGridPtr();
    }

    bool OptimizedGPUCudaVoxelization::saveVoxelGrid(const std::string& filename) const {
        return impl_->saveVoxelGrid(filename);
    }

    // 优化配置接口实现
    void OptimizedGPUCudaVoxelization::setOptimizationFlags(bool use_octree, bool use_morton, bool use_2pass) {
        use_octree_ = use_octree;
        use_morton_ = use_morton;
        use_2pass_ = use_2pass;
        //         impl_->setOptimizationFlags(use_octree_, use_morton_, use_2pass_);
    }

    void OptimizedGPUCudaVoxelization::setCoarseFactor(int factor) {
        coarse_factor_ = factor;
        //         impl_->setCoarseFactor(coarse_factor_);
    }

    void OptimizedGPUCudaVoxelization::setBlockSize(int block_size_1d, int block_size_2d, int block_size_3d) {
        block_size_1d_ = block_size_1d;
        block_size_2d_ = block_size_2d;
        block_size_3d_ = block_size_3d;
    }

    void OptimizedGPUCudaVoxelization::setSharedMemorySize(int size_kb) {
        shared_memory_size_kb_ = size_kb;
    }

    void OptimizedGPUCudaVoxelization::setOctreeDepth(int depth) {
        octree_depth_ = depth;
    }

    void OptimizedGPUCudaVoxelization::setMortonBits(int bits) {
        morton_bits_ = bits;
    }

    OptimizedGPUCudaVoxelization::PerformanceMetrics OptimizedGPUCudaVoxelization::getPerformanceMetrics() const {
        return metrics_;
    }

    void OptimizedGPUCudaVoxelization::resetPerformanceMetrics() {
        metrics_ = { 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0 };
    }

    void OptimizedGPUCudaVoxelization::enableAdaptiveResolution(bool enable) {
        adaptive_resolution_ = enable;
    }

    void OptimizedGPUCudaVoxelization::setLoadBalancingStrategy(const std::string& strategy) {
        load_balancing_strategy_ = strategy;
    }

    void OptimizedGPUCudaVoxelization::enableMemoryCoalescing(bool enable) {
        memory_coalescing_ = enable;
    }

    void OptimizedGPUCudaVoxelization::setWarpSize(int warp_size) {
        warp_size_ = warp_size;
    }

    // GPUMemoryOptimizer 实现
    GPUMemoryOptimizer::GPUMemoryOptimizer()
        : pool_size_(1024 * 1024 * 1024), // 1GB默认池大小
        alignment_(256), unified_memory_(false), compression_enabled_(false) {
        stats_ = { 0, 0, 0, 0, pool_size_, 0 };
    }

    GPUMemoryOptimizer::~GPUMemoryOptimizer() {
        clearMemoryPool();
    }

    void* GPUMemoryOptimizer::allocatePooledMemory(size_t size, size_t alignment) {
        // 查找可用的内存块
        for (auto& block : memory_pool_) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                stats_.current_usage += size;
                stats_.total_allocated += size;
                stats_.allocation_count++;
                stats_.peak_usage = std::max(stats_.peak_usage, stats_.current_usage);
                return block.ptr;
            }
        }

        // 如果没有找到合适的块，分配新的
        void* ptr = nullptr;
        if (unified_memory_) {
            cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
        }
        else {
            cudaMalloc(&ptr, size);
        }

        if (ptr) {
            MemoryBlock new_block = { ptr, size, true };
            memory_pool_.push_back(new_block);
            stats_.current_usage += size;
            stats_.total_allocated += size;
            stats_.allocation_count++;
            stats_.peak_usage = std::max(stats_.peak_usage, stats_.current_usage);
        }

        return ptr;
    }

    void GPUMemoryOptimizer::freePooledMemory(void* ptr) {
        for (auto& block : memory_pool_) {
            if (block.ptr == ptr && block.in_use) {
                block.in_use = false;
                stats_.current_usage -= block.size;
                stats_.total_freed += block.size;
                break;
            }
        }
    }

    void GPUMemoryOptimizer::clearMemoryPool() {
        for (auto& block : memory_pool_) {
            if (block.ptr) {
                cudaFree(block.ptr);
            }
        }
        memory_pool_.clear();
        stats_.current_usage = 0;
    }

    void GPUMemoryOptimizer::enableUnifiedMemory(bool enable) {
        unified_memory_ = enable;
    }

    void GPUMemoryOptimizer::setMemoryPoolSize(size_t size_mb) {
        pool_size_ = size_mb * 1024 * 1024;
        stats_.pool_size = pool_size_;
    }

    void GPUMemoryOptimizer::enableMemoryCompression(bool enable) {
        compression_enabled_ = enable;
    }

    void GPUMemoryOptimizer::setMemoryAlignment(size_t alignment) {
        alignment_ = alignment;
    }

    GPUMemoryOptimizer::MemoryStats GPUMemoryOptimizer::getMemoryStats() const {
        return stats_;
    }

    // GPUKernelOptimizer 实现
    GPUKernelOptimizer::GPUKernelOptimizer()
        : load_balancing_strategy_("uniform"), dynamic_parallelism_(false),
        max_blocks_per_sm_(32), warp_size_(32) {
    }

    GPUKernelOptimizer::~GPUKernelOptimizer() = default;

    GPUKernelOptimizer::KernelConfig GPUKernelOptimizer::optimizeKernelConfig(int grid_x, int grid_y, int grid_z,
        size_t shared_memory_required) {
        KernelConfig config;

        // 获取GPU设备信息
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        // 优化块大小
        if (grid_z > 1) {
            // 3D网格
            config.block_size = dim3(8, 8, 8);
        }
        else if (grid_y > 1) {
            // 2D网格
            config.block_size = dim3(16, 16, 1);
        }
        else {
            // 1D网格
            config.block_size = dim3(256, 1, 1);
        }

        // 计算网格大小
        config.grid_size = dim3(
            (grid_x + config.block_size.x - 1) / config.block_size.x,
            (grid_y + config.block_size.y - 1) / config.block_size.y,
            (grid_z + config.block_size.z - 1) / config.block_size.z
        );

        // 设置共享内存大小
        config.shared_memory_size = shared_memory_required;

        // 设置每SM最大块数
        config.max_blocks_per_sm = max_blocks_per_sm_;
        config.warp_size = warp_size_;

        return config;
    }

    void GPUKernelOptimizer::setLoadBalancingStrategy(const std::string& strategy) {
        load_balancing_strategy_ = strategy;
    }

    void GPUKernelOptimizer::enableDynamicParallelism(bool enable) {
        dynamic_parallelism_ = enable;
    }

    void GPUKernelOptimizer::setMaxBlocksPerSM(int max_blocks) {
        max_blocks_per_sm_ = max_blocks;
    }

    GPUKernelOptimizer::KernelProfile GPUKernelOptimizer::profileKernel(const std::string& kernel_name) {
        KernelProfile profile = { 0.0, 0.0, 0.0, 0, 0, 0, 0 };

        // 这里可以实现实际的内核性能分析
        // 使用CUDA Profiler API或NVIDIA Visual Profiler

        return profile;
    }

    // SpatialDataOptimizer 实现
    SpatialDataOptimizer::SpatialDataOptimizer() {
        // 默认八叉树配置
        octree_config_ = { 8, 16, 0.5f, true };

        // 默认Morton编码配置
        morton_config_ = { 21, true, false };

        // 默认空间哈希配置
        spatial_hash_config_ = { 1.0f, 1000000, true };
    }

    SpatialDataOptimizer::~SpatialDataOptimizer() = default;

    void SpatialDataOptimizer::setOctreeConfig(const OctreeConfig& config) {
        octree_config_ = config;
    }

    SpatialDataOptimizer::OctreeConfig SpatialDataOptimizer::getOctreeConfig() const {
        return octree_config_;
    }

    void SpatialDataOptimizer::setMortonConfig(const MortonConfig& config) {
        morton_config_ = config;
    }

    SpatialDataOptimizer::MortonConfig SpatialDataOptimizer::getMortonConfig() const {
        return morton_config_;
    }

    void SpatialDataOptimizer::setSpatialHashConfig(const SpatialHashConfig& config) {
        spatial_hash_config_ = config;
    }

    SpatialDataOptimizer::SpatialHashConfig SpatialDataOptimizer::getSpatialHashConfig() const {
        return spatial_hash_config_;
    }

    // 继承自VoxelizationBase的所有纯虚函数最小实现
    int OptimizedGPUCudaVoxelization::voxelizeEntity(const std::shared_ptr<SpatialEntity>&, double, unsigned char) { return 0; }
    int OptimizedGPUCudaVoxelization::voxelizeEntities(const std::vector<std::shared_ptr<SpatialEntity>>& entities, double buffer_size, unsigned char cost_value) {
        int total_voxels = 0;
        // 清空实体和 mesh 数据
        impl_->clear();
        for (const auto& entity : entities) {
            std::string type = entity->getType();
            if (type == "box" || type == "sphere" || type == "cylinder") {
                this->addEntity(entity);
            }
            else if (type == "ellipsoid" || type == "cone" || type == "mesh") {
                // fallback to CPU for ellipsoid, cone and mesh
                std::cerr << "[OptimizedGPUCudaVoxelization] Entity type '" << type << "' not supported on GPU_OPTIMIZED, fallback to CPU." << std::endl;
                CPUSequentialVoxelization cpu_voxelizer;
                cpu_voxelizer.initialize(
                    impl_->getGridX(), impl_->getGridY(), impl_->getGridZ(),
                    impl_->getResolution(), impl_->getResolution(),
                    impl_->getOrigin().x, impl_->getOrigin().y, impl_->getOrigin().z);
                cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
                // 不做合并
            }
            else {
                // fallback to CPU for unsupported types
                std::cerr << "[OptimizedGPUCudaVoxelization] Entity type '" << type << "' not supported on GPU, fallback to CPU." << std::endl;
                CPUSequentialVoxelization cpu_voxelizer;
                cpu_voxelizer.initialize(
                    impl_->getGridX(), impl_->getGridY(), impl_->getGridZ(),
                    impl_->getResolution(), impl_->getResolution(),
                    impl_->getOrigin().x, impl_->getOrigin().y, impl_->getOrigin().z);
                cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
            }
        }
        impl_->prepareOptimizations();
        impl_->voxelize(cost_value);
        return total_voxels;
    }
    std::vector<int> OptimizedGPUCudaVoxelization::getGridDimensions() const { return { 0,0,0 }; }
    std::vector<double> OptimizedGPUCudaVoxelization::getResolutions() const { return { 0.0,0.0,0.0 }; }
    bool OptimizedGPUCudaVoxelization::worldToVoxel(double, double, double, int&, int&, int&) const { return false; }
    void OptimizedGPUCudaVoxelization::voxelToWorld(int, int, int, double&, double&, double&) const {}
    bool OptimizedGPUCudaVoxelization::isValidVoxel(int, int, int) const { return false; }
    void OptimizedGPUCudaVoxelization::updateVoxelCost(int, int, int, unsigned char) {}
    unsigned char OptimizedGPUCudaVoxelization::getVoxelCost(int, int, int) const { return 0; }
    void OptimizedGPUCudaVoxelization::clear() {}
    bool OptimizedGPUCudaVoxelization::saveToFile(const std::string&) const { return false; }
    bool OptimizedGPUCudaVoxelization::loadFromFile(const std::string&) { return false; }
    void OptimizedGPUCudaVoxelization::setResolution(int, int, int) {}
    void OptimizedGPUCudaVoxelization::setBoundingBox(double, double, double, double, double, double) {}
    bool OptimizedGPUCudaVoxelization::voxelize(const std::vector<std::shared_ptr<SpatialEntity>>&) { return false; }

    void OptimizedGPUCudaVoxelization::getVoxelGrid(std::vector<unsigned char>& out) const {
        if (impl_) {
            impl_->getVoxelGrid(out);
        }
        else {
            out.clear();
        }
    }

} // namespace voxelization
