#include "gpu_voxelization_optimized.hpp"
#include "gpu_voxelization.hpp"
#include "spatial_entities.hpp"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>

using namespace voxelization;

class PerformanceBenchmark {
private:
    std::vector<std::shared_ptr<SpatialEntity>> test_entities_;
    int grid_x_, grid_y_, grid_z_;
    double resolution_;

public:
    PerformanceBenchmark(int grid_x = 256, int grid_y = 256, int grid_z = 64, double resolution = 0.1)
        : grid_x_(grid_x), grid_y_(grid_y), grid_z_(grid_z), resolution_(resolution) {
        generateTestEntities();
    }

    void generateTestEntities() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> pos_dist(-50.0, 50.0);
        std::uniform_real_distribution<double> size_dist(1.0, 10.0);
        std::uniform_int_distribution<int> type_dist(0, 2);

        // 生成100个随机实体
        for (int i = 0; i < 100; ++i) {
            std::shared_ptr<SpatialEntity> entity;

            switch (type_dist(gen)) {
            case 0: // Box
                entity = std::make_shared<BoxEntity>(
                    pos_dist(gen), pos_dist(gen), pos_dist(gen),
                    size_dist(gen), size_dist(gen), size_dist(gen)
                );
                break;
            case 1: // Sphere
                entity = std::make_shared<SphereEntity>(
                    pos_dist(gen), pos_dist(gen), pos_dist(gen),
                    size_dist(gen)
                );
                break;
            case 2: // Cylinder
                entity = std::make_shared<CylinderEntity>(
                    pos_dist(gen), pos_dist(gen), pos_dist(gen),
                    size_dist(gen), size_dist(gen) * 2
                );
                break;
            }

            test_entities_.push_back(entity);
        }
    }

    struct BenchmarkResult {
        std::string method_name;
        double total_time_ms;
        double voxelization_time_ms;
        size_t memory_usage_bytes;
        int active_voxels;
        double throughput_mvoxels_per_sec;
        std::map<std::string, double> detailed_metrics;
    };

    BenchmarkResult benchmarkBasicGPU() {
        std::cout << "Running Basic GPU benchmark..." << std::endl;

        GPUCudaVoxelization gpu_vox;
        gpu_vox.initialize(grid_x_, grid_y_, grid_z_, resolution_, resolution_, -50.0, -50.0, -50.0);

        auto start_time = std::chrono::high_resolution_clock::now();
        gpu_vox.voxelize(test_entities_);
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::vector<unsigned char> voxel_grid;
        gpu_vox.getVoxelGrid(voxel_grid);

        BenchmarkResult result;
        result.method_name = "Basic GPU";
        result.total_time_ms = duration.count() / 1000.0;
        result.voxelization_time_ms = result.total_time_ms;
        result.memory_usage_bytes = voxel_grid.size();
        result.active_voxels = std::count_if(voxel_grid.begin(), voxel_grid.end(), [](unsigned char v) { return v > 0; });
        result.throughput_mvoxels_per_sec = (grid_x_ * grid_y_ * grid_z_) / (result.total_time_ms / 1000.0) / 1000000.0;

        return result;
    }

    BenchmarkResult benchmarkOptimizedGPU(bool use_octree = true, bool use_morton = true, bool use_2pass = true) {
        std::string method_name = "Optimized GPU";
        if (use_octree) method_name += " + Octree";
        if (use_morton) method_name += " + Morton";
        if (use_2pass) method_name += " + 2Pass";

        std::cout << "Running " << method_name << " benchmark..." << std::endl;

        OptimizedGPUCudaVoxelization opt_gpu_vox;
        opt_gpu_vox.setOptimizationFlags(use_octree, use_morton, use_2pass);
        opt_gpu_vox.initialize(grid_x_, grid_y_, grid_z_, resolution_, resolution_, -50.0, -50.0, -50.0);

        auto start_time = std::chrono::high_resolution_clock::now();
        opt_gpu_vox.voxelize(test_entities_);
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::vector<unsigned char> voxel_grid;
        opt_gpu_vox.getVoxelGrid(voxel_grid);
        auto metrics = opt_gpu_vox.getPerformanceMetrics();

        BenchmarkResult result;
        result.method_name = method_name;
        result.total_time_ms = duration.count() / 1000.0;
        result.voxelization_time_ms = metrics.fine_pass_time + metrics.coarse_pass_time;
        result.memory_usage_bytes = voxel_grid.size();
        result.active_voxels = std::count_if(voxel_grid.begin(), voxel_grid.end(), [](unsigned char v) { return v > 0; });
        result.throughput_mvoxels_per_sec = (grid_x_ * grid_y_ * grid_z_) / (result.total_time_ms / 1000.0) / 1000000.0;

        // 详细指标
        result.detailed_metrics["octree_build_time"] = metrics.octree_build_time;
        result.detailed_metrics["morton_generation_time"] = metrics.morton_generation_time;
        result.detailed_metrics["coarse_pass_time"] = metrics.coarse_pass_time;
        result.detailed_metrics["fine_pass_time"] = metrics.fine_pass_time;

        return result;
    }

    void runComprehensiveBenchmark() {
        std::vector<BenchmarkResult> results;

        // 基础GPU
        results.push_back(benchmarkBasicGPU());

        // 不同优化组合
        results.push_back(benchmarkOptimizedGPU(true, true, true));   // 全优化
        results.push_back(benchmarkOptimizedGPU(true, false, false)); // 仅八叉树
        results.push_back(benchmarkOptimizedGPU(false, true, false)); // 仅Morton
        results.push_back(benchmarkOptimizedGPU(false, false, true)); // 仅2Pass
        results.push_back(benchmarkOptimizedGPU(true, true, false));  // 八叉树+Morton
        results.push_back(benchmarkOptimizedGPU(true, false, true));  // 八叉树+2Pass
        results.push_back(benchmarkOptimizedGPU(false, true, true));  // Morton+2Pass

        printResults(results);
        saveResults(results);
    }

    void printResults(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "GPU VOXELIZATION OPTIMIZATION BENCHMARK RESULTS" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        std::cout << "Grid Size: " << grid_x_ << "x" << grid_y_ << "x" << grid_z_
            << " (" << (grid_x_ * grid_y_ * grid_z_) << " voxels)" << std::endl;
        std::cout << "Resolution: " << resolution_ << " m" << std::endl;
        std::cout << "Test Entities: " << test_entities_.size() << std::endl;
        std::cout << std::endl;

        // 表头
        std::cout << std::left << std::setw(30) << "Method"
            << std::setw(12) << "Time (ms)"
            << std::setw(12) << "Throughput"
            << std::setw(12) << "Active Voxels"
            << std::setw(15) << "Memory (MB)"
            << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        // 结果
        for (const auto& result : results) {
            std::cout << std::left << std::setw(30) << result.method_name
                << std::setw(12) << std::fixed << std::setprecision(2) << result.total_time_ms
                << std::setw(12) << std::fixed << std::setprecision(2) << result.throughput_mvoxels_per_sec
                << std::setw(12) << result.active_voxels
                << std::setw(15) << std::fixed << std::setprecision(2) << (result.memory_usage_bytes / 1024.0 / 1024.0)
                << std::endl;
        }

        // 详细指标
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "DETAILED OPTIMIZATION METRICS" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        for (const auto& result : results) {
            if (!result.detailed_metrics.empty()) {
                std::cout << "\n" << result.method_name << ":" << std::endl;
                for (const auto& [key, value] : result.detailed_metrics) {
                    std::cout << "  " << std::left << std::setw(25) << key
                        << ": " << std::fixed << std::setprecision(3) << value << " ms" << std::endl;
                }
            }
        }

        // 性能提升分析
        if (results.size() > 1) {
            std::cout << "\n" << std::string(80, '=') << std::endl;
            std::cout << "PERFORMANCE IMPROVEMENT ANALYSIS" << std::endl;
            std::cout << std::string(80, '=') << std::endl;

            double baseline_time = results[0].total_time_ms;
            for (size_t i = 1; i < results.size(); ++i) {
                double improvement = (baseline_time - results[i].total_time_ms) / baseline_time * 100.0;
                double speedup = baseline_time / results[i].total_time_ms;

                std::cout << results[i].method_name << " vs " << results[0].method_name << ":" << std::endl;
                std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
                std::cout << "  Improvement: " << std::fixed << std::setprecision(1) << improvement << "%" << std::endl;
                std::cout << std::endl;
            }
        }
    }

    void saveResults(const std::vector<BenchmarkResult>& results) {
        std::ofstream file("gpu_optimization_results.csv");
        if (file.is_open()) {
            // 写入表头
            file << "Method,Total Time (ms),Voxelization Time (ms),Throughput (MVoxels/s),"
                << "Active Voxels,Memory Usage (MB),Octree Build Time (ms),"
                << "Morton Generation Time (ms),Coarse Pass Time (ms),Fine Pass Time (ms)\n";

            // 写入数据
            for (const auto& result : results) {
                file << result.method_name << ","
                    << result.total_time_ms << ","
                    << result.voxelization_time_ms << ","
                    << result.throughput_mvoxels_per_sec << ","
                    << result.active_voxels << ","
                    << (result.memory_usage_bytes / 1024.0 / 1024.0) << ","
                    << result.detailed_metrics.at("octree_build_time") << ","
                    << result.detailed_metrics.at("morton_generation_time") << ","
                    << result.detailed_metrics.at("coarse_pass_time") << ","
                    << result.detailed_metrics.at("fine_pass_time") << "\n";
            }

            file.close();
            std::cout << "\nResults saved to: gpu_optimization_results.csv" << std::endl;
        }
    }

    void testMemoryOptimization() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "MEMORY OPTIMIZATION TEST" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        GPUMemoryOptimizer mem_optimizer;

        // 测试内存池
        mem_optimizer.setMemoryPoolSize(512); // 512MB
        mem_optimizer.enableUnifiedMemory(true);

        void* ptr1 = mem_optimizer.allocatePooledMemory(1024 * 1024); // 1MB
        void* ptr2 = mem_optimizer.allocatePooledMemory(2048 * 1024); // 2MB
        void* ptr3 = mem_optimizer.allocatePooledMemory(512 * 1024);  // 512KB

        auto stats = mem_optimizer.getMemoryStats();
        std::cout << "Memory Pool Statistics:" << std::endl;
        std::cout << "  Total Allocated: " << stats.total_allocated / 1024 / 1024 << " MB" << std::endl;
        std::cout << "  Current Usage: " << stats.current_usage / 1024 / 1024 << " MB" << std::endl;
        std::cout << "  Peak Usage: " << stats.peak_usage / 1024 / 1024 << " MB" << std::endl;
        std::cout << "  Allocation Count: " << stats.allocation_count << std::endl;

        mem_optimizer.freePooledMemory(ptr1);
        mem_optimizer.freePooledMemory(ptr2);
        mem_optimizer.freePooledMemory(ptr3);
    }

    void testKernelOptimization() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "KERNEL OPTIMIZATION TEST" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        GPUKernelOptimizer kernel_optimizer;

        // 测试不同网格大小的内核配置
        std::vector<std::tuple<int, int, int>> grid_sizes = {
            {256, 256, 64},
            {512, 512, 128},
            {1024, 1024, 256}
        };

        for (const auto& [x, y, z] : grid_sizes) {
            auto config = kernel_optimizer.optimizeKernelConfig(x, y, z, 48 * 1024);

            std::cout << "Grid " << x << "x" << y << "x" << z << ":" << std::endl;
            std::cout << "  Block Size: " << config.block_size.x << "x" << config.block_size.y << "x" << config.block_size.z << std::endl;
            std::cout << "  Grid Size: " << config.grid_size.x << "x" << config.grid_size.y << "x" << config.grid_size.z << std::endl;
            std::cout << "  Shared Memory: " << config.shared_memory_size / 1024 << " KB" << std::endl;
            std::cout << "  Max Blocks per SM: " << config.max_blocks_per_sm << std::endl;
            std::cout << std::endl;
        }
    }
};

int main() {
    try {
        std::cout << "GPU Voxelization Optimization Benchmark" << std::endl;
        std::cout << "======================================" << std::endl;

        // 检查CUDA可用性
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            std::cerr << "CUDA is not available. Cannot run GPU benchmarks." << std::endl;
            return 1;
        }

        std::cout << "CUDA devices found: " << deviceCount << std::endl;

        // 获取GPU信息
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Using GPU: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Global Memory: " << prop.totalGlobalMem / 1024 / 1024 / 1024 << " GB" << std::endl;
        std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
        std::cout << std::endl;

        // 运行基准测试
        PerformanceBenchmark benchmark(256, 256, 64, 0.1);

        // 综合性能测试
        benchmark.runComprehensiveBenchmark();

        // 内存优化测试
        benchmark.testMemoryOptimization();

        // 内核优化测试
        benchmark.testKernelOptimization();

        std::cout << "\nBenchmark completed successfully!" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error during benchmark: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
