#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>

#include "voxelization_factory.hpp"
#include "spatial_entities.hpp"

using namespace voxelization;
using voxelization::VoxelizationFactory;
using AlgorithmType = VoxelizationFactory::AlgorithmType;

class GPUPerformanceAnalyzer {
private:
    std::vector<std::shared_ptr<SpatialEntity>> test_entities_;
    int grid_x_, grid_y_, grid_z_;
    double resolution_xy_, resolution_z_;
    double origin_x_, origin_y_, origin_z_;

public:
    GPUPerformanceAnalyzer() {
        // Initialize with a reasonable test case
        grid_x_ = 100;
        grid_y_ = 100;
        grid_z_ = 50;
        resolution_xy_ = 0.1;
        resolution_z_ = 0.1;
        origin_x_ = -5.0;
        origin_y_ = -5.0;
        origin_z_ = -2.5;

        generateTestEntities();
    }

    void generateTestEntities() {
        test_entities_.clear();

        // Generate a mix of entities
        for (int i = 0; i < 10; ++i) {
            // Box
            auto box = std::make_shared<BoxEntity>(
                -3.0 + i * 0.6, -3.0 + i * 0.6, -1.0 + i * 0.2,
                0.5, 0.5, 0.5
            );
            test_entities_.push_back(box);

            // Sphere
            auto sphere = std::make_shared<SphereEntity>(
                3.0 - i * 0.6, 3.0 - i * 0.6, 1.0 - i * 0.2, 0.3
            );
            test_entities_.push_back(sphere);

            // Cylinder
            auto cylinder = std::make_shared<CylinderEntity>(
                -2.0 + i * 0.4, 2.0 - i * 0.4, 0.0, 0.2, 0.8
            );
            test_entities_.push_back(cylinder);
        }
    }

    void analyzeGPUMemoryTransfer() {
        std::cout << "\n=== GPU Memory Transfer Analysis ===" << std::endl;

        auto gpu_voxelizer = VoxelizationFactory::createAlgorithm(AlgorithmType::GPU_CUDA);
        gpu_voxelizer->initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);

        // Measure GPU memory allocation time
        auto start = std::chrono::high_resolution_clock::now();
        gpu_voxelizer->voxelizeEntities(test_entities_);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Total GPU voxelization time: " << duration.count() / 1000.0 << " ms" << std::endl;

        // Calculate memory transfer overhead
        size_t grid_size = grid_x_ * grid_y_ * grid_z_;
        size_t transfer_size = grid_size * sizeof(unsigned char);
        double transfer_time_estimate = transfer_size / (12.0 * 1024.0 * 1024.0 * 1024.0) * 1000.0; // PCIe 3.0 x16

        std::cout << "Grid size: " << grid_x_ << "x" << grid_y_ << "x" << grid_z_ << " = " << grid_size << " voxels" << std::endl;
        std::cout << "Transfer size: " << transfer_size / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Estimated transfer time (PCIe 3.0 x16): " << transfer_time_estimate << " ms" << std::endl;
    }

    void analyzeKernelPerformance() {
        std::cout << "\n=== Kernel Performance Analysis ===" << std::endl;

        // Test different block sizes
        std::vector<std::pair<int, int>> block_configs = {
            {8, 8}, {16, 16}, {32, 32}, {8, 16}, {16, 8}, {32, 16}, {16, 32}
        };

        for (const auto& config : block_configs) {
            int block_x = config.first;
            int block_y = config.second;
            int block_z = 4; // Keep Z dimension small

            std::cout << "\nTesting block size: " << block_x << "x" << block_y << "x" << block_z << std::endl;

            auto gpu_voxelizer = VoxelizationFactory::createAlgorithm(AlgorithmType::GPU_CUDA);
            gpu_voxelizer->initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);

            // Warm up
            gpu_voxelizer->voxelizeEntities(test_entities_);

            // Measure performance
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 10; ++i) {
                gpu_voxelizer->voxelizeEntities(test_entities_);
            }
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double avg_time = duration.count() / 10000.0; // Average over 10 runs

            std::cout << "  Average time: " << avg_time << " ms" << std::endl;

            // Calculate occupancy
            int threads_per_block = block_x * block_y * block_z;
            int max_blocks_per_sm = 32; // GTX 1080 Ti has 28 SMs
            int max_threads_per_sm = 2048;
            int blocks_per_sm = max_threads_per_sm / threads_per_block;
            double occupancy = (double)(threads_per_block * blocks_per_sm) / max_threads_per_sm;

            std::cout << "  Threads per block: " << threads_per_block << std::endl;
            std::cout << "  Estimated occupancy: " << occupancy * 100 << "%" << std::endl;
        }
    }

    void compareWithCPU() {
        std::cout << "\n=== CPU vs GPU Comparison ===" << std::endl;

        // CPU Sequential
        auto cpu_seq = VoxelizationFactory::createAlgorithm(AlgorithmType::CPU_SEQUENTIAL);
        cpu_seq->initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);

        auto start = std::chrono::high_resolution_clock::now();
        cpu_seq->voxelizeEntities(test_entities_);
        auto end = std::chrono::high_resolution_clock::now();

        auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // CPU Parallel
        auto cpu_par = VoxelizationFactory::createAlgorithm(AlgorithmType::CPU_PARALLEL);
        cpu_par->initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);

        start = std::chrono::high_resolution_clock::now();
        cpu_par->voxelizeEntities(test_entities_);
        end = std::chrono::high_resolution_clock::now();

        auto cpu_par_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // GPU
        auto gpu = VoxelizationFactory::createAlgorithm(AlgorithmType::GPU_CUDA);
        gpu->initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);

        start = std::chrono::high_resolution_clock::now();
        gpu->voxelizeEntities(test_entities_);
        end = std::chrono::high_resolution_clock::now();

        auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "CPU Sequential: " << cpu_time.count() / 1000.0 << " ms" << std::endl;
        std::cout << "CPU Parallel:   " << cpu_par_time.count() / 1000.0 << " ms" << std::endl;
        std::cout << "GPU CUDA:       " << gpu_time.count() / 1000.0 << " ms" << std::endl;

        std::cout << "\nSpeedup:" << std::endl;
        std::cout << "GPU vs CPU Sequential: " << (double)cpu_time.count() / gpu_time.count() << "x" << std::endl;
        std::cout << "GPU vs CPU Parallel:   " << (double)cpu_par_time.count() / gpu_time.count() << "x" << std::endl;
    }

    void analyzeScalability() {
        std::cout << "\n=== Scalability Analysis ===" << std::endl;

        std::vector<int> grid_sizes = { 50, 100, 200, 400 };

        for (int size : grid_sizes) {
            std::cout << "\nGrid size: " << size << "x" << size << "x" << (size / 2) << std::endl;

            // Update grid dimensions
            grid_x_ = grid_y_ = size;
            grid_z_ = size / 2;

            // CPU Sequential
            auto cpu_seq = VoxelizationFactory::createAlgorithm(AlgorithmType::CPU_SEQUENTIAL);
            cpu_seq->initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);

            auto start = std::chrono::high_resolution_clock::now();
            cpu_seq->voxelizeEntities(test_entities_);
            auto end = std::chrono::high_resolution_clock::now();

            auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            // GPU
            auto gpu = VoxelizationFactory::createAlgorithm(AlgorithmType::GPU_CUDA);
            gpu->initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);

            start = std::chrono::high_resolution_clock::now();
            gpu->voxelizeEntities(test_entities_);
            end = std::chrono::high_resolution_clock::now();

            auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cout << "  CPU: " << cpu_time.count() / 1000.0 << " ms" << std::endl;
            std::cout << "  GPU: " << gpu_time.count() / 1000.0 << " ms" << std::endl;
            std::cout << "  Speedup: " << (double)cpu_time.count() / gpu_time.count() << "x" << std::endl;
        }
    }

    void runFullAnalysis() {
        std::cout << "=== GPU Performance Analysis Tool ===" << std::endl;
        std::cout << "Test configuration:" << std::endl;
        std::cout << "  Grid: " << grid_x_ << "x" << grid_y_ << "x" << grid_z_ << std::endl;
        std::cout << "  Resolution: " << resolution_xy_ << " m" << std::endl;
        std::cout << "  Entities: " << test_entities_.size() << std::endl;

        analyzeGPUMemoryTransfer();
        analyzeKernelPerformance();
        compareWithCPU();
        analyzeScalability();

        std::cout << "\n=== Analysis Complete ===" << std::endl;
    }
};

int main() {
    try {
        GPUPerformanceAnalyzer analyzer;
        analyzer.runFullAnalysis();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
