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
#include <map>
#include <functional>

#include "voxelization_factory.hpp"
#include "spatial_entities.hpp"

using namespace voxelization;

/**
 * @brief Advanced benchmark result structure
 */
struct AdvancedBenchmarkResult {
    std::string algorithm_name;
    std::string test_scenario;
    double total_time_ms;
    double avg_time_per_entity_ms;
    int total_voxels;
    double voxels_per_second;
    int num_entities;
    double memory_usage_mb;
    double cpu_utilization;
    int grid_size_x, grid_size_y, grid_size_z;
    double resolution_xy, resolution_z;
};

/**
 * @brief Test scenario configuration
 */
struct TestScenario {
    std::string name;
    int num_entities;
    int grid_x, grid_y, grid_z;
    double resolution_xy, resolution_z;
    double entity_min_size, entity_max_size;
    double entity_min_pos, entity_max_pos;
    std::string description;
};

/**
 * @brief Advanced random entity generator with more control
 */
class AdvancedRandomEntityGenerator {
private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<double> pos_dist_;
    std::uniform_real_distribution<double> size_dist_;
    std::uniform_real_distribution<double> radius_dist_;
    std::uniform_int_distribution<int> type_dist_;
    std::uniform_real_distribution<double> ratio_dist_;

public:
    AdvancedRandomEntityGenerator(double min_pos, double max_pos,
        double min_size, double max_size)
        : gen_(rd_()),
        pos_dist_(min_pos, max_pos),
        size_dist_(min_size, max_size),
        radius_dist_(min_size / 2, max_size / 2),
        type_dist_(0, 4),
        ratio_dist_(0.5, 2.0) {
    }

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

    // Generate entities with specific distribution patterns
    std::vector<std::shared_ptr<SpatialEntity>> generateClusteredEntities(int count, int clusters = 5) {
        std::vector<std::shared_ptr<SpatialEntity>> entities;
        entities.reserve(count);

        std::uniform_real_distribution<double> cluster_center_dist(-40.0, 40.0);
        std::uniform_real_distribution<double> cluster_radius_dist(5.0, 15.0);

        for (int cluster = 0; cluster < clusters; ++cluster) {
            double center_x = cluster_center_dist(gen_);
            double center_y = cluster_center_dist(gen_);
            double center_z = cluster_center_dist(gen_);
            double cluster_radius = cluster_radius_dist(gen_);

            int entities_in_cluster = count / clusters + (cluster < count % clusters ? 1 : 0);

            for (int i = 0; i < entities_in_cluster; ++i) {
                std::uniform_real_distribution<double> local_pos_dist(-cluster_radius, cluster_radius);
                double x = center_x + local_pos_dist(gen_);
                double y = center_y + local_pos_dist(gen_);
                double z = center_z + local_pos_dist(gen_);

                int type = type_dist_(gen_);
                switch (type) {
                case 0: {
                    double size_x = size_dist_(gen_);
                    double size_y = size_dist_(gen_);
                    double size_z = size_dist_(gen_);
                    entities.push_back(std::make_shared<BoxEntity>(x, y, z, size_x, size_y, size_z));
                    break;
                }
                case 1: {
                    double radius = radius_dist_(gen_);
                    double height = size_dist_(gen_);
                    entities.push_back(std::make_shared<CylinderEntity>(x, y, z, radius, height));
                    break;
                }
                case 2: {
                    double radius = radius_dist_(gen_);
                    entities.push_back(std::make_shared<SphereEntity>(x, y, z, radius));
                    break;
                }
                case 3: {
                    double radius_x = radius_dist_(gen_);
                    double radius_y = radius_dist_(gen_);
                    double radius_z = radius_dist_(gen_);
                    entities.push_back(std::make_shared<EllipsoidEntity>(x, y, z, radius_x, radius_y, radius_z));
                    break;
                }
                case 4: {
                    double radius = radius_dist_(gen_);
                    double height = size_dist_(gen_);
                    entities.push_back(std::make_shared<ConeEntity>(x, y, z, radius, height));
                    break;
                }
                }
            }
        }

        return entities;
    }
};

/**
 * @brief Advanced performance benchmark class
 */
class AdvancedPerformanceBenchmark {
private:
    std::vector<TestScenario> test_scenarios_;
    std::map<std::string, std::vector<AdvancedBenchmarkResult>> all_results_;

public:
    AdvancedPerformanceBenchmark() {
        initializeTestScenarios();
    }

    void initializeTestScenarios() {
        // Small scale test
        test_scenarios_.push_back({
            "Small Scale",
            50,
            100, 100, 50,
            0.2, 0.2,
            1.0, 5.0,
            -25.0, 25.0,
            "Small number of entities with coarse resolution"
            });

        // Medium scale test
        test_scenarios_.push_back({
            "Medium Scale",
            200,
            200, 200, 100,
            0.1, 0.1,
            1.0, 8.0,
            -50.0, 50.0,
            "Medium number of entities with medium resolution"
            });

        // Large scale test
        test_scenarios_.push_back({
            "Large Scale",
            500,
            400, 400, 200,
            0.05, 0.05,
            0.5, 10.0,
            -100.0, 100.0,
            "Large number of entities with fine resolution"
            });

        // High density test
        test_scenarios_.push_back({
            "High Density",
            300,
            300, 300, 150,
            0.08, 0.08,
            2.0, 6.0,
            -30.0, 30.0,
            "High density of entities in small space"
            });

        // Clustered test
        test_scenarios_.push_back({
            "Clustered",
            400,
            350, 350, 175,
            0.1, 0.1,
            1.0, 7.0,
            -80.0, 80.0,
            "Entities clustered in groups"
            });
    }

    AdvancedBenchmarkResult runSingleBenchmark(const TestScenario& scenario,
        VoxelizationFactory::AlgorithmType algorithm_type,
        bool use_clustered = false) {
        auto algorithm = VoxelizationFactory::createAlgorithm(algorithm_type);
        std::string algorithm_name = VoxelizationFactory::getAlgorithmName(algorithm_type);

        std::cout << "Running " << algorithm_name << " on " << scenario.name << " scenario..." << std::endl;

        // Generate entities
        AdvancedRandomEntityGenerator generator(scenario.entity_min_pos, scenario.entity_max_pos,
            scenario.entity_min_size, scenario.entity_max_size);

        std::vector<std::shared_ptr<SpatialEntity>> entities;
        if (use_clustered && scenario.name == "Clustered") {
            entities = generator.generateClusteredEntities(scenario.num_entities);
        }
        else {
            entities = generator.generateEntityBatch(scenario.num_entities);
        }

        // Initialize voxel grid
        algorithm->initialize(scenario.grid_x, scenario.grid_y, scenario.grid_z,
            scenario.resolution_xy, scenario.resolution_z);

        // Measure memory usage before
        double memory_before = getCurrentMemoryUsage(scenario);

        // Run voxelization and measure time
        auto start_time = std::chrono::high_resolution_clock::now();

        int total_voxels = algorithm->voxelizeEntities(entities, 0.1, 255);

        auto end_time = std::chrono::high_resolution_clock::now();

        // Measure memory usage after
        double memory_after = getCurrentMemoryUsage(scenario);

        // Calculate timing
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double total_time_ms = duration.count() / 1000.0;
        double avg_time_per_entity_ms = total_time_ms / entities.size();
        double voxels_per_second = (total_voxels * 1000.0) / total_time_ms;

        AdvancedBenchmarkResult result;
        result.algorithm_name = algorithm_name;
        result.test_scenario = scenario.name;
        result.total_time_ms = total_time_ms;
        result.avg_time_per_entity_ms = avg_time_per_entity_ms;
        result.total_voxels = total_voxels;
        result.voxels_per_second = voxels_per_second;
        result.num_entities = entities.size();
        result.memory_usage_mb = memory_after - memory_before;
        result.cpu_utilization = estimateCPUUtilization(algorithm_type);
        result.grid_size_x = scenario.grid_x;
        result.grid_size_y = scenario.grid_y;
        result.grid_size_z = scenario.grid_z;
        result.resolution_xy = scenario.resolution_xy;
        result.resolution_z = scenario.resolution_z;

        return result;
    }

    void runAllBenchmarks() {
        std::vector<VoxelizationFactory::AlgorithmType> algorithms = {
            VoxelizationFactory::AlgorithmType::CPU_SEQUENTIAL,
            VoxelizationFactory::AlgorithmType::CPU_PARALLEL,
            VoxelizationFactory::AlgorithmType::GPU_CUDA
        };

        printSystemInfo();

        for (const auto& scenario : test_scenarios_) {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "Testing Scenario: " << scenario.name << std::endl;
            std::cout << "Description: " << scenario.description << std::endl;
            std::cout << std::string(60, '=') << std::endl;

            std::vector<AdvancedBenchmarkResult> scenario_results;

            for (auto algorithm : algorithms) {
                try {
                    bool use_clustered = (scenario.name == "Clustered");
                    AdvancedBenchmarkResult result = runSingleBenchmark(scenario, algorithm, use_clustered);
                    scenario_results.push_back(result);
                }
                catch (const std::exception& e) {
                    std::cout << "Error running benchmark for "
                        << VoxelizationFactory::getAlgorithmName(algorithm)
                        << ": " << e.what() << std::endl;
                }
            }

            printScenarioResults(scenario_results);
            all_results_[scenario.name] = scenario_results;
        }

        printOverallAnalysis();
        saveDetailedResults();
    }

private:
    void printSystemInfo() {
        std::cout << "\n=== Advanced Performance Benchmark System Information ===" << std::endl;
        std::cout << "Number of test scenarios: " << test_scenarios_.size() << std::endl;

#ifdef _OPENMP
        std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
#else
        std::cout << "OpenMP: Not available" << std::endl;
#endif

        std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
    }

    void printScenarioResults(const std::vector<AdvancedBenchmarkResult>& results) {
        std::cout << "\nResults for scenario: " << results[0].test_scenario << std::endl;
        std::cout << std::setw(20) << "Algorithm"
            << std::setw(15) << "Time(ms)"
            << std::setw(20) << "Avg/Entity(ms)"
            << std::setw(15) << "Voxels"
            << std::setw(20) << "Voxels/sec"
            << std::setw(15) << "Memory(MB)" << std::endl;
        std::cout << std::string(105, '-') << std::endl;

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
            std::cout << "\nSpeedup Analysis:" << std::endl;
            double baseline_time = results[0].total_time_ms;

            for (size_t i = 1; i < results.size(); ++i) {
                double speedup = baseline_time / results[i].total_time_ms;
                std::cout << "  " << results[i].algorithm_name << " vs " << results[0].algorithm_name
                    << " speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            }
        }
    }

    void printOverallAnalysis() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "OVERALL PERFORMANCE ANALYSIS" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        // Find best performing algorithm for each scenario
        for (const auto& scenario_pair : all_results_) {
            const std::string& scenario_name = scenario_pair.first;
            const std::vector<AdvancedBenchmarkResult>& results = scenario_pair.second;

            if (results.empty()) continue;

            auto fastest = std::min_element(results.begin(), results.end(),
                [](const AdvancedBenchmarkResult& a, const AdvancedBenchmarkResult& b) {
                    return a.total_time_ms < b.total_time_ms;
                });

            std::cout << "\nBest performer for " << scenario_name << ": "
                << fastest->algorithm_name
                << " (" << std::fixed << std::setprecision(2) << fastest->total_time_ms << " ms)" << std::endl;
        }

        // Calculate average speedup across all scenarios
        std::map<std::string, double> avg_speedup;
        std::map<std::string, int> scenario_count;

        for (const auto& scenario_pair : all_results_) {
            const std::vector<AdvancedBenchmarkResult>& results = scenario_pair.second;
            if (results.size() < 2) continue;

            double baseline_time = results[0].total_time_ms;
            for (size_t i = 1; i < results.size(); ++i) {
                double speedup = baseline_time / results[i].total_time_ms;
                avg_speedup[results[i].algorithm_name] += speedup;
                scenario_count[results[i].algorithm_name]++;
            }
        }

        std::cout << "\nAverage Speedup Across All Scenarios:" << std::endl;
        for (const auto& speedup_pair : avg_speedup) {
            double avg = speedup_pair.second / scenario_count[speedup_pair.first];
            std::cout << "  " << speedup_pair.first << ": "
                << std::fixed << std::setprecision(2) << avg << "x" << std::endl;
        }
    }

    void saveDetailedResults() {
        // Save CSV results
        std::ofstream csv_file("advanced_benchmark_results.csv");
        if (!csv_file.is_open()) {
            std::cout << "Warning: Could not open advanced_benchmark_results.csv for writing" << std::endl;
            return;
        }

        csv_file << "Scenario,Algorithm,Total_Time_ms,Avg_Time_per_Entity_ms,Total_Voxels,"
            << "Voxels_per_Second,Memory_MB,CPU_Utilization,Grid_Size_X,Grid_Size_Y,Grid_Size_Z,"
            << "Resolution_XY,Resolution_Z\n";

        for (const auto& scenario_pair : all_results_) {
            for (const auto& result : scenario_pair.second) {
                csv_file << result.test_scenario << ","
                    << result.algorithm_name << ","
                    << result.total_time_ms << ","
                    << result.avg_time_per_entity_ms << ","
                    << result.total_voxels << ","
                    << result.voxels_per_second << ","
                    << result.memory_usage_mb << ","
                    << result.cpu_utilization << ","
                    << result.grid_size_x << ","
                    << result.grid_size_y << ","
                    << result.grid_size_z << ","
                    << result.resolution_xy << ","
                    << result.resolution_z << "\n";
            }
        }

        csv_file.close();

        // Save summary report
        std::ofstream report_file("benchmark_summary_report.txt");
        if (report_file.is_open()) {
            report_file << "Advanced Voxelization Performance Benchmark Report\n";
            report_file << "==================================================\n\n";

            report_file << "Test Scenarios:\n";
            for (const auto& scenario : test_scenarios_) {
                report_file << "- " << scenario.name << ": " << scenario.description << "\n";
            }

            report_file << "\nDetailed Results:\n";
            for (const auto& scenario_pair : all_results_) {
                report_file << "\n" << scenario_pair.first << ":\n";
                for (const auto& result : scenario_pair.second) {
                    report_file << "  " << result.algorithm_name << ": "
                        << result.total_time_ms << " ms, "
                        << result.voxels_per_second << " voxels/sec\n";
                }
            }

            report_file.close();
        }

        std::cout << "\nDetailed results saved to:" << std::endl;
        std::cout << "  - advanced_benchmark_results.csv" << std::endl;
        std::cout << "  - benchmark_summary_report.txt" << std::endl;
    }

    double getCurrentMemoryUsage(const TestScenario& scenario) {
        double grid_memory_mb = (scenario.grid_x * scenario.grid_y * scenario.grid_z * sizeof(unsigned char)) / (1024.0 * 1024.0);
        return grid_memory_mb;
    }

    double estimateCPUUtilization(VoxelizationFactory::AlgorithmType algorithm_type) {
        switch (algorithm_type) {
        case VoxelizationFactory::AlgorithmType::CPU_SEQUENTIAL:
            return 100.0; // Single core utilization
        case VoxelizationFactory::AlgorithmType::CPU_PARALLEL:
#ifdef _OPENMP
            return 100.0 * omp_get_max_threads() / std::thread::hardware_concurrency();
#else
            return 100.0;
#endif
        case VoxelizationFactory::AlgorithmType::GPU_CUDA:
            return 10.0; // GPU offload, low CPU usage
        default:
            return 50.0;
        }
    }
};

/**
 * @brief Main function
 */
int main(int argc, char* argv[]) {
    std::cout << "=== Advanced Voxelization Performance Benchmark ===" << std::endl;

    try {
        AdvancedPerformanceBenchmark benchmark;
        benchmark.runAllBenchmarks();

        std::cout << "\nAdvanced benchmark completed successfully!" << std::endl;
        std::cout << "Check the generated CSV and report files for detailed analysis." << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error during advanced benchmark: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
