#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <thread>
#include <omp.h>

#include "voxelization_factory.hpp"
#include "spatial_entities.hpp"

using namespace voxelization;

/**
 * @brief Performance benchmark results structure
 */
struct BenchmarkResult {
    std::string algorithm_name;
    double total_time_ms;
    double avg_time_per_entity_ms;
    int total_voxels;
    double voxels_per_second;
    int num_entities;
    double memory_usage_mb;
};

/**
 * @brief Random entity generator
 */
class RandomEntityGenerator {
private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<double> pos_dist_;
    std::uniform_real_distribution<double> size_dist_;
    std::uniform_real_distribution<double> radius_dist_;
    std::uniform_int_distribution<int> type_dist_;

public:
    RandomEntityGenerator(double min_pos, double max_pos, double min_size, double max_size)
        : gen_(rd_()),
        pos_dist_(min_pos, max_pos),
        size_dist_(min_size, max_size),
        radius_dist_(min_size / 2, max_size / 2),
        type_dist_(0, 4) {
    } // 0-4 for 5 entity types

    std::shared_ptr<SpatialEntity> generateRandomEntity() {
        int type = type_dist_(gen_);
        double x = pos_dist_(gen_);
        double y = pos_dist_(gen_);
        double z = pos_dist_(gen_);

        switch (type) {
        case 0: { // Box
            double size_x = size_dist_(gen_);
            double size_y = size_dist_(gen_);
            double size_z = size_dist_(gen_);
            return std::make_shared<BoxEntity>(x, y, z, size_x, size_y, size_z);
        }
        case 1: { // Cylinder
            double radius = radius_dist_(gen_);
            double height = size_dist_(gen_);
            return std::make_shared<CylinderEntity>(x, y, z, radius, height);
        }
        case 2: { // Sphere
            double radius = radius_dist_(gen_);
            return std::make_shared<SphereEntity>(x, y, z, radius);
        }
        case 3: { // Ellipsoid
            double radius_x = radius_dist_(gen_);
            double radius_y = radius_dist_(gen_);
            double radius_z = radius_dist_(gen_);
            return std::make_shared<EllipsoidEntity>(x, y, z, radius_x, radius_y, radius_z);
        }
        case 4: { // Cone
            double radius = radius_dist_(gen_);
            double height = size_dist_(gen_);
            return std::make_shared<ConeEntity>(x, y, z, radius, height);
        }
        default:
            return std::make_shared<SphereEntity>(x, y, z, radius_dist_(gen_));
        }
    }

    std::vector<std::shared_ptr<SpatialEntity>> generateEntityBatch(int count) {
        std::vector<std::shared_ptr<SpatialEntity>> entities;
        entities.reserve(count);

        for (int i = 0; i < count; ++i) {
            entities.push_back(generateRandomEntity());
        }

        return entities;
    }
};

/**
 * @brief Performance benchmark class
 */
class PerformanceBenchmark {
private:
    std::vector<std::shared_ptr<SpatialEntity>> test_entities_;
    int grid_x_, grid_y_, grid_z_;
    double resolution_xy_, resolution_z_;
    double origin_x_, origin_y_, origin_z_;

public:
    PerformanceBenchmark(int num_entities = 100,
        int grid_x = 200, int grid_y = 200, int grid_z = 100,
        double resolution_xy = 0.1, double resolution_z = 0.1,
        double origin_x = 0.0, double origin_y = 0.0, double origin_z = 0.0)
        : grid_x_(grid_x), grid_y_(grid_y), grid_z_(grid_z),
        resolution_xy_(resolution_xy), resolution_z_(resolution_z),
        origin_x_(origin_x), origin_y_(origin_y), origin_z_(origin_z) {

        // Generate random entities
        RandomEntityGenerator generator(-50.0, 50.0, 1.0, 10.0);
        test_entities_ = generator.generateEntityBatch(num_entities);

        std::cout << "Generated " << num_entities << " random entities for testing." << std::endl;
    }

    BenchmarkResult runBenchmark(VoxelizationFactory::AlgorithmType algorithm_type) {
        auto algorithm = VoxelizationFactory::createAlgorithm(algorithm_type);
        std::string algorithm_name = VoxelizationFactory::getAlgorithmName(algorithm_type);

        std::cout << "\nRunning benchmark for: " << algorithm_name << std::endl;

        // Initialize voxel grid
        algorithm->initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);

        // Measure memory usage before
        double memory_before = getCurrentMemoryUsage();

        // Run voxelization and measure time
        auto start_time = std::chrono::high_resolution_clock::now();

        int total_voxels = algorithm->voxelizeEntities(test_entities_, 0.1, 255);

        auto end_time = std::chrono::high_resolution_clock::now();

        // Measure memory usage after
        double memory_after = getCurrentMemoryUsage();

        // Calculate timing
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double total_time_ms = duration.count() / 1000.0;
        double avg_time_per_entity_ms = total_time_ms / test_entities_.size();
        double voxels_per_second = (total_voxels * 1000.0) / total_time_ms;

        BenchmarkResult result;
        result.algorithm_name = algorithm_name;
        result.total_time_ms = total_time_ms;
        result.avg_time_per_entity_ms = avg_time_per_entity_ms;
        result.total_voxels = total_voxels;
        result.voxels_per_second = voxels_per_second;
        result.num_entities = test_entities_.size();
        result.memory_usage_mb = memory_after - memory_before;

        return result;
    }

    void runAllBenchmarks() {
        std::vector<VoxelizationFactory::AlgorithmType> algorithms = {
            VoxelizationFactory::AlgorithmType::CPU_SEQUENTIAL,
            VoxelizationFactory::AlgorithmType::CPU_PARALLEL,
            VoxelizationFactory::AlgorithmType::GPU_CUDA
        };

        std::vector<BenchmarkResult> results;

        // Print system information
        printSystemInfo();

        // Run benchmarks
        for (auto algorithm : algorithms) {
            try {
                BenchmarkResult result = runBenchmark(algorithm);
                results.push_back(result);
            }
            catch (const std::exception& e) {
                std::cout << "Error running benchmark for "
                    << VoxelizationFactory::getAlgorithmName(algorithm)
                    << ": " << e.what() << std::endl;
            }
        }

        // Print results
        printResults(results);

        // Save results to file
        saveResultsToFile(results);
    }

private:
    void printSystemInfo() {
        std::cout << "\n=== System Information ===" << std::endl;
        std::cout << "Grid size: " << grid_x_ << "x" << grid_y_ << "x" << grid_z_ << std::endl;
        std::cout << "Resolution: " << resolution_xy_ << "x" << resolution_xy_ << "x" << resolution_z_ << std::endl;
        std::cout << "Origin: (" << origin_x_ << ", " << origin_y_ << ", " << origin_z_ << ")" << std::endl;

#ifdef _OPENMP
        std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
#else
        std::cout << "OpenMP: Not available" << std::endl;
#endif

        std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
    }

    void printResults(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n=== Performance Benchmark Results ===" << std::endl;
        std::cout << std::setw(20) << "Algorithm"
            << std::setw(15) << "Total Time(ms)"
            << std::setw(20) << "Avg/Entity(ms)"
            << std::setw(15) << "Total Voxels"
            << std::setw(20) << "Voxels/sec"
            << std::setw(15) << "Memory(MB)" << std::endl;
        std::cout << std::string(100, '-') << std::endl;

        for (const auto& result : results) {
            std::cout << std::setw(20) << result.algorithm_name
                << std::setw(15) << std::fixed << std::setprecision(2) << result.total_time_ms
                << std::setw(20) << std::fixed << std::setprecision(4) << result.avg_time_per_entity_ms
                << std::setw(15) << result.total_voxels
                << std::setw(20) << std::fixed << std::setprecision(0) << result.voxels_per_second
                << std::setw(15) << std::fixed << std::setprecision(2) << result.memory_usage_mb << std::endl;
        }

        // Calculate speedup
        if (results.size() >= 2) {
            std::cout << "\n=== Speedup Analysis ===" << std::endl;
            double baseline_time = results[0].total_time_ms;

            for (size_t i = 1; i < results.size(); ++i) {
                double speedup = baseline_time / results[i].total_time_ms;
                std::cout << results[i].algorithm_name << " vs " << results[0].algorithm_name
                    << " speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            }
        }
    }

    void saveResultsToFile(const std::vector<BenchmarkResult>& results) {
        std::ofstream file("benchmark_results.csv");
        if (!file.is_open()) {
            std::cout << "Warning: Could not open benchmark_results.csv for writing" << std::endl;
            return;
        }

        file << "Algorithm,Total_Time_ms,Avg_Time_per_Entity_ms,Total_Voxels,Voxels_per_Second,Memory_MB\n";

        for (const auto& result : results) {
            file << result.algorithm_name << ","
                << result.total_time_ms << ","
                << result.avg_time_per_entity_ms << ","
                << result.total_voxels << ","
                << result.voxels_per_second << ","
                << result.memory_usage_mb << "\n";
        }

        file.close();
        std::cout << "\nResults saved to benchmark_results.csv" << std::endl;
    }

    double getCurrentMemoryUsage() {
        // Simple memory usage estimation
        // In a real implementation, you might want to use platform-specific APIs
        // For now, we'll return a rough estimate based on grid size
        double grid_memory_mb = (grid_x_ * grid_y_ * grid_z_ * sizeof(unsigned char)) / (1024.0 * 1024.0);
        return grid_memory_mb;
    }
};

/**
 * @brief Main function
 */
int main(int argc, char* argv[]) {
    std::cout << "=== Voxelization Performance Benchmark ===" << std::endl;

    // Parse command line arguments
    int num_entities = 100;
    int grid_x = 200, grid_y = 200, grid_z = 100;
    double resolution_xy = 0.1, resolution_z = 0.1;

    if (argc > 1) num_entities = std::stoi(argv[1]);
    if (argc > 2) grid_x = std::stoi(argv[2]);
    if (argc > 3) grid_y = std::stoi(argv[3]);
    if (argc > 4) grid_z = std::stoi(argv[4]);
    if (argc > 5) resolution_xy = std::stod(argv[5]);
    if (argc > 6) resolution_z = std::stod(argv[6]);

    std::cout << "Parameters:" << std::endl;
    std::cout << "  Number of entities: " << num_entities << std::endl;
    std::cout << "  Grid size: " << grid_x << "x" << grid_y << "x" << grid_z << std::endl;
    std::cout << "  Resolution: " << resolution_xy << "x" << resolution_xy << "x" << resolution_z << std::endl;

    try {
        PerformanceBenchmark benchmark(num_entities, grid_x, grid_y, grid_z,
            resolution_xy, resolution_z);
        benchmark.runAllBenchmarks();

        std::cout << "\nBenchmark completed successfully!" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error during benchmark: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
