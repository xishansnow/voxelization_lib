#include "voxelization_factory.hpp"
#include "spatial_entities.hpp"
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <map>

#ifdef ENABLE_OCTOMAP
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#endif

using namespace voxelization;

/**
 * @brief Display voxel grid statistics
 */
void displayVoxelStatistics(const std::vector<unsigned char>& voxel_grid, int grid_x, int grid_y, int grid_z) {
    std::cout << "\n=== Voxel Grid Statistics ===" << std::endl;

    int total_voxels = voxel_grid.size();
    int active_voxels = std::count_if(voxel_grid.begin(), voxel_grid.end(), [](unsigned char v) { return v > 0; });
    double occupancy_rate = (double)active_voxels / total_voxels * 100.0;

    std::cout << "   - Total voxels: " << total_voxels << std::endl;
    std::cout << "   - Active voxels: " << active_voxels << std::endl;
    std::cout << "   - Occupancy rate: " << std::fixed << std::setprecision(2) << occupancy_rate << "%" << std::endl;

    // Count voxels by cost value
    std::map<unsigned char, int> cost_distribution;
    for (unsigned char cost : voxel_grid) {
        if (cost > 0) {
            cost_distribution[cost]++;
        }
    }

    std::cout << "   - Cost value distribution:" << std::endl;
    for (const auto& pair : cost_distribution) {
        unsigned char cost = pair.first;
        int count = pair.second;
        std::cout << "     * Cost " << static_cast<int>(cost) << ": " << count << " voxels" << std::endl;
    }
}

/**
 * @brief Display 2D slice of voxel grid
 */
void displayVoxelSlice(const std::vector<unsigned char>& voxel_grid, int grid_x, int grid_y, int grid_z,
    int slice_z, const std::string& slice_name) {
    std::cout << "\n=== " << slice_name << " (Z=" << slice_z << ") ===" << std::endl;

    if (slice_z < 0 || slice_z >= grid_z) {
        std::cout << "   - Invalid Z slice: " << slice_z << std::endl;
        return;
    }

    // Display header
    std::cout << "   ";
    for (int x = 0; x < std::min(grid_x, 20); ++x) {
        std::cout << std::setw(2) << x % 10;
    }
    std::cout << std::endl;

    // Display grid content
    for (int y = 0; y < std::min(grid_y, 20); ++y) {
        std::cout << std::setw(2) << y << " ";
        for (int x = 0; x < std::min(grid_x, 20); ++x) {
            int index = slice_z * grid_x * grid_y + y * grid_x + x;
            if (index < voxel_grid.size()) {
                unsigned char cost = voxel_grid[index];
                if (cost > 0) {
                    std::cout << "██";
                }
                else {
                    std::cout << "  ";
                }
            }
            else {
                std::cout << "  ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "   Legend: ██ = occupied voxel,   = empty voxel" << std::endl;
}

/**
 * @brief Export voxel grid to PGM format for visualization
 */
bool exportVoxelGridToPGM(const std::vector<unsigned char>& voxel_grid, int grid_x, int grid_y, int grid_z,
    const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << "   - ERROR: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    // PGM header
    file << "P2" << std::endl;
    file << "# Voxel Grid Export" << std::endl;
    file << grid_x << " " << grid_y << std::endl;
    file << "255" << std::endl;

    // Export middle Z slice
    int middle_z = grid_z / 2;
    for (int y = 0; y < grid_y; ++y) {
        for (int x = 0; x < grid_x; ++x) {
            int index = middle_z * grid_x * grid_y + y * grid_x + x;
            if (index < voxel_grid.size()) {
                file << static_cast<int>(voxel_grid[index]) << " ";
            }
            else {
                file << "0 ";
            }
        }
        file << std::endl;
    }

    file.close();
    std::cout << "   - Voxel grid exported to " << filename << " (Z=" << middle_z << " slice)" << std::endl;
    return true;
}

/**
 * @brief Export voxel grid to CSV format for analysis
 */
bool exportVoxelGridToCSV(const std::vector<unsigned char>& voxel_grid, int grid_x, int grid_y, int grid_z,
    const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << "   - ERROR: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    // CSV header
    file << "x,y,z,cost" << std::endl;

    // Export all active voxels
    for (int z = 0; z < grid_z; ++z) {
        for (int y = 0; y < grid_y; ++y) {
            for (int x = 0; x < grid_x; ++x) {
                int index = z * grid_x * grid_y + y * grid_x + x;
                if (index < voxel_grid.size() && voxel_grid[index] > 0) {
                    file << x << "," << y << "," << z << "," << static_cast<int>(voxel_grid[index]) << std::endl;
                }
            }
        }
    }

    file.close();
    std::cout << "   - Active voxels exported to " << filename << std::endl;
    return true;
}

/**
 * @brief Display entity information in voxel grid
 */
void displayEntityInfo(const std::vector<std::shared_ptr<SpatialEntity>>& entities,
    const std::shared_ptr<VoxelizationBase>& voxelizer) {
    std::cout << "\n=== Entity Information ===" << std::endl;

    for (size_t i = 0; i < entities.size(); ++i) {
        const auto& entity = entities[i];
        auto properties = entity->getProperties();

        std::cout << "   Entity " << (i + 1) << " (" << entity->getType() << "):" << std::endl;

        if (entity->getType() == "box") {
            std::cout << "     - Center: (" << properties["center_x"] << ", "
                << properties["center_y"] << ", " << properties["center_z"] << ")" << std::endl;
            std::cout << "     - Size: (" << properties["size_x"] << ", "
                << properties["size_y"] << ", " << properties["size_z"] << ")" << std::endl;
        }
        else if (entity->getType() == "sphere") {
            std::cout << "     - Center: (" << properties["center_x"] << ", "
                << properties["center_y"] << ", " << properties["center_z"] << ")" << std::endl;
            std::cout << "     - Radius: " << properties["radius"] << std::endl;
        }
        else if (entity->getType() == "cylinder") {
            std::cout << "     - Center: (" << properties["center_x"] << ", "
                << properties["center_y"] << ", " << properties["center_z"] << ")" << std::endl;
            std::cout << "     - Radius: " << properties["radius"] << std::endl;
            std::cout << "     - Height: " << properties["height"] << std::endl;
        }

        // Convert entity center to voxel coordinates
        int voxel_x, voxel_y, voxel_z;
        if (voxelizer->worldToVoxel(properties["center_x"], properties["center_y"], properties["center_z"],
            voxel_x, voxel_y, voxel_z)) {
            std::cout << "     - Voxel center: (" << voxel_x << ", " << voxel_y << ", " << voxel_z << ")" << std::endl;
        }
    }
}

/**
 * @brief Export voxel grid to OctoMap .bt format
 */
#ifdef ENABLE_OCTOMAP
bool exportVoxelGridToBT(const std::vector<unsigned char>& voxel_grid, int grid_x, int grid_y, int grid_z,
    double resolution, double origin_x, double origin_y, double origin_z,
    const std::string& filename) {
    try {
        octomap::OcTree tree(resolution);

        // Convert voxel grid to OctoMap
        for (int z = 0; z < grid_z; ++z) {
            for (int y = 0; y < grid_y; ++y) {
                for (int x = 0; x < grid_x; ++x) {
                    int index = z * grid_x * grid_y + y * grid_x + x;
                    if (index < voxel_grid.size() && voxel_grid[index] > 0) {
                        // Convert voxel coordinates to world coordinates
                        double world_x = origin_x + x * resolution;
                        double world_y = origin_y + y * resolution;
                        double world_z = origin_z + z * resolution;

                        // Add occupied node to OctoMap
                        tree.updateNode(octomap::point3d(world_x, world_y, world_z), true);
                    }
                }
            }
        }

        // Update inner occupancy for proper tree structure
        tree.updateInnerOccupancy();

        // Write to .bt file
        if (tree.writeBinary(filename)) {
            std::cout << "   - Voxel grid exported to " << filename << " (.bt format)" << std::endl;
            std::cout << "     * Resolution: " << resolution << "m" << std::endl;
            std::cout << "     * Tree depth: " << tree.getTreeDepth() << std::endl;
            std::cout << "     * Memory usage: " << tree.memoryUsage() << " bytes" << std::endl;
            return true;
        }
        else {
            std::cout << "   - ERROR: Failed to write .bt file: " << filename << std::endl;
            return false;
        }
    }
    catch (const std::exception& e) {
        std::cout << "   - ERROR: Exception during .bt export: " << e.what() << std::endl;
        return false;
    }
}
#else
bool exportVoxelGridToBT(const std::vector<unsigned char>& voxel_grid, int grid_x, int grid_y, int grid_z,
    double resolution, double origin_x, double origin_y, double origin_z,
    const std::string& filename) {
    std::cout << "   - WARNING: OctoMap not available - .bt export disabled" << std::endl;
    return false;
}
#endif

int main() {
    std::cout << "=== Voxelization Library Test ===" << std::endl;

    // Test 1: Create different spatial entities
    std::cout << "\n1. Creating spatial entities..." << std::endl;

    auto box_entity = std::make_shared<BoxEntity>(5.0, 5.0, 5.0, 2.0, 2.0, 2.0);
    auto sphere_entity = std::make_shared<SphereEntity>(8.0, 5.0, 5.0, 1.5);
    auto cylinder_entity = std::make_shared<CylinderEntity>(2.0, 8.0, 5.0, 1.0, 3.0);

    std::cout << "   - Box entity created at (5, 5, 5) with size (2, 2, 2)" << std::endl;
    std::cout << "   - Sphere entity created at (8, 5, 5) with radius 1.5" << std::endl;
    std::cout << "   - Cylinder entity created at (2, 8, 5) with radius 1.0 and height 3.0" << std::endl;

    // Test 2: Test entity properties
    std::cout << "\n2. Testing entity properties..." << std::endl;

    auto box_bbox = box_entity->getBoundingBox();
    std::cout << "   - Box bounding box: ["
        << box_bbox[0] << ", " << box_bbox[1] << ", " << box_bbox[2] << "] to ["
        << box_bbox[3] << ", " << box_bbox[4] << ", " << box_bbox[5] << "]" << std::endl;

    std::cout << "   - Testing point inside box (5, 5, 5): "
        << (box_entity->isPointInside(5.0, 5.0, 5.0) ? "INSIDE" : "OUTSIDE") << std::endl;
    std::cout << "   - Testing point outside box (10, 10, 10): "
        << (box_entity->isPointInside(10.0, 10.0, 10.0) ? "INSIDE" : "OUTSIDE") << std::endl;

    // Test 3: Create voxelization algorithms
    std::cout << "\n3. Creating voxelization algorithms..." << std::endl;

    auto cpu_seq = VoxelizationFactory::createAlgorithm(VoxelizationFactory::AlgorithmType::CPU_SEQUENTIAL);
    auto cpu_par = VoxelizationFactory::createAlgorithm(VoxelizationFactory::AlgorithmType::CPU_PARALLEL);

    if (!cpu_seq || !cpu_par) {
        std::cout << "   ERROR: Failed to create voxelization algorithms!" << std::endl;
        return -1;
    }

    std::cout << "   - CPU Sequential algorithm created" << std::endl;
    std::cout << "   - CPU Parallel algorithm created" << std::endl;

    // Test 4: Initialize voxelization
    std::cout << "\n4. Initializing voxelization..." << std::endl;

    int grid_x = 100, grid_y = 100, grid_z = 50;
    double resolution_xy = 0.1, resolution_z = 0.1;
    double origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;

    cpu_seq->initialize(grid_x, grid_y, grid_z, resolution_xy, resolution_z, origin_x, origin_y, origin_z);
    cpu_par->initialize(grid_x, grid_y, grid_z, resolution_xy, resolution_z, origin_x, origin_y, origin_z);

    std::cout << "   - Grid size: " << grid_x << "x" << grid_y << "x" << grid_z << std::endl;
    std::cout << "   - Resolution: " << resolution_xy << "m (xy), " << resolution_z << "m (z)" << std::endl;
    std::cout << "   - Origin: (" << origin_x << ", " << origin_y << ", " << origin_z << ")" << std::endl;

    // Test 5: Voxelize single entity
    std::cout << "\n5. Voxelizing single entity..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    int marked_voxels = cpu_seq->voxelizeEntity(box_entity, 0.0, 255);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "   - CPU Sequential: " << marked_voxels << " voxels marked in "
        << duration.count() << " microseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    marked_voxels = cpu_par->voxelizeEntity(box_entity, 0.0, 255);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "   - CPU Parallel: " << marked_voxels << " voxels marked in "
        << duration.count() << " microseconds" << std::endl;

    // Test 6: Voxelize multiple entities
    std::cout << "\n6. Voxelizing multiple entities..." << std::endl;

    std::vector<std::shared_ptr<SpatialEntity>> entities = { box_entity, sphere_entity, cylinder_entity };

    cpu_seq->clear();
    start = std::chrono::high_resolution_clock::now();
    marked_voxels = cpu_seq->voxelizeEntities(entities, 0.0, 255);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "   - CPU Sequential: " << marked_voxels << " voxels marked in "
        << duration.count() << " microseconds" << std::endl;

    cpu_par->clear();
    start = std::chrono::high_resolution_clock::now();
    marked_voxels = cpu_par->voxelizeEntities(entities, 0.0, 255);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "   - CPU Parallel: " << marked_voxels << " voxels marked in "
        << duration.count() << " microseconds" << std::endl;

    // Test 6.5: Display voxelization results
    std::cout << "\n6.5. Displaying voxelization results..." << std::endl;

    // Get voxel grid from CPU sequential algorithm
    const unsigned char* voxel_data = cpu_seq->getVoxelGrid();
    std::vector<unsigned char> voxel_grid(voxel_data, voxel_data + grid_x * grid_y * grid_z);

    // Display entity information
    displayEntityInfo(entities, cpu_seq);

    // Display voxel statistics
    displayVoxelStatistics(voxel_grid, grid_x, grid_y, grid_z);

    // Display 2D slices at different Z levels
    displayVoxelSlice(voxel_grid, grid_x, grid_y, grid_z, 0, "Bottom Slice");
    displayVoxelSlice(voxel_grid, grid_x, grid_y, grid_z, grid_z / 4, "Quarter Height Slice");
    displayVoxelSlice(voxel_grid, grid_x, grid_y, grid_z, grid_z / 2, "Middle Slice");
    displayVoxelSlice(voxel_grid, grid_x, grid_y, grid_z, 3 * grid_z / 4, "Three Quarter Height Slice");
    displayVoxelSlice(voxel_grid, grid_x, grid_y, grid_z, grid_z - 1, "Top Slice");

    // Export voxel grid for visualization
    std::cout << "\n6.6. Exporting voxel grid for visualization..." << std::endl;
    exportVoxelGridToPGM(voxel_grid, grid_x, grid_y, grid_z, "voxel_grid_middle_slice.pgm");
    exportVoxelGridToCSV(voxel_grid, grid_x, grid_y, grid_z, "active_voxels.csv");
    exportVoxelGridToBT(voxel_grid, grid_x, grid_y, grid_z, resolution_xy, origin_x, origin_y, origin_z, "cpu_seq_voxel_grid.bt");

    // Test 7: Test coordinate conversion
    std::cout << "\n7. Testing coordinate conversion..." << std::endl;

    int voxel_x, voxel_y, voxel_z;
    double world_x, world_y, world_z;

    // World to voxel
    if (cpu_seq->worldToVoxel(5.0, 5.0, 5.0, voxel_x, voxel_y, voxel_z)) {
        std::cout << "   - World (5, 5, 5) -> Voxel (" << voxel_x << ", " << voxel_y << ", " << voxel_z << ")" << std::endl;
    }

    // Voxel to world
    cpu_seq->voxelToWorld(50, 50, 25, world_x, world_y, world_z);
    std::cout << "   - Voxel (50, 50, 25) -> World (" << world_x << ", " << world_y << ", " << world_z << ")" << std::endl;

    // Test 8: Test voxel access
    std::cout << "\n8. Testing voxel access..." << std::endl;

    if (cpu_seq->isValidVoxel(50, 50, 25)) {
        unsigned char cost = cpu_seq->getVoxelCost(50, 50, 25);
        std::cout << "   - Voxel (50, 50, 25) cost: " << static_cast<int>(cost) << std::endl;
    }

    // Test 9: Test file I/O
    std::cout << "\n9. Testing file I/O..." << std::endl;

    if (cpu_seq->saveToFile("test_voxel_grid.bin")) {
        std::cout << "   - Voxel grid saved to test_voxel_grid.bin" << std::endl;
    }
    else {
        std::cout << "   - ERROR: Failed to save voxel grid" << std::endl;
    }

    // Test 10: Available algorithms
    std::cout << "\n10. Available algorithms:" << std::endl;
    auto algorithms = VoxelizationFactory::getAvailableAlgorithms();
    for (const auto& alg : algorithms) {
        std::cout << "   - " << alg << std::endl;
    }

    // === GPU_CUDA 测试 ===
    std::cout << "\n=== Testing GPU_CUDA ===" << std::endl;
    auto gpu_cuda = VoxelizationFactory::createAlgorithm(VoxelizationFactory::AlgorithmType::GPU_CUDA);
    if (gpu_cuda) {
        gpu_cuda->initialize(grid_x, grid_y, grid_z, resolution_xy, resolution_z, origin_x, origin_y, origin_z);
        gpu_cuda->clear();
        auto start = std::chrono::high_resolution_clock::now();
        int marked_voxels = gpu_cuda->voxelizeEntities(entities, 0.0, 255);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "   - GPU_CUDA: " << marked_voxels << " voxels marked in " << duration.count() << " microseconds" << std::endl;
        const unsigned char* voxel_data = gpu_cuda->getVoxelGrid();
        std::vector<unsigned char> voxel_grid(voxel_data, voxel_data + grid_x * grid_y * grid_z);
        displayVoxelStatistics(voxel_grid, grid_x, grid_y, grid_z);
        displayVoxelSlice(voxel_grid, grid_x, grid_y, grid_z, grid_z / 2, "GPU_CUDA Middle Slice");
        exportVoxelGridToPGM(voxel_grid, grid_x, grid_y, grid_z, "gpu_cuda_voxel_grid_middle_slice.pgm");
        exportVoxelGridToCSV(voxel_grid, grid_x, grid_y, grid_z, "gpu_cuda_active_voxels.csv");
        exportVoxelGridToBT(voxel_grid, grid_x, grid_y, grid_z, resolution_xy, origin_x, origin_y, origin_z, "gpu_cuda_voxel_grid.bt");
    }
    else {
        std::cout << "   - ERROR: Failed to create GPU_CUDA algorithm instance!" << std::endl;
    }

    // === GPU_OPTIMIZED 测试 ===
    std::cout << "\n=== Testing GPU_OPTIMIZED ===" << std::endl;
    auto gpu_opt = VoxelizationFactory::createAlgorithm(VoxelizationFactory::AlgorithmType::GPU_OPTIMIZED);
    if (gpu_opt) {
        gpu_opt->initialize(grid_x, grid_y, grid_z, resolution_xy, resolution_z, origin_x, origin_y, origin_z);
        gpu_opt->clear();
        auto start = std::chrono::high_resolution_clock::now();
        int marked_voxels = gpu_opt->voxelizeEntities(entities, 0.0, 255);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "   - GPU_OPTIMIZED: " << marked_voxels << " voxels marked in " << duration.count() << " microseconds" << std::endl;
        const unsigned char* voxel_data = gpu_opt->getVoxelGrid();
        std::vector<unsigned char> voxel_grid(voxel_data, voxel_data + grid_x * grid_y * grid_z);
        displayVoxelStatistics(voxel_grid, grid_x, grid_y, grid_z);
        displayVoxelSlice(voxel_grid, grid_x, grid_y, grid_z, grid_z / 2, "GPU_OPTIMIZED Middle Slice");
        exportVoxelGridToPGM(voxel_grid, grid_x, grid_y, grid_z, "gpu_optimized_voxel_grid_middle_slice.pgm");
        exportVoxelGridToCSV(voxel_grid, grid_x, grid_y, grid_z, "gpu_optimized_active_voxels.csv");
        exportVoxelGridToBT(voxel_grid, grid_x, grid_y, grid_z, resolution_xy, origin_x, origin_y, origin_z, "gpu_optimized_voxel_grid.bt");
    }
    else {
        std::cout << "   - ERROR: Failed to create GPU_OPTIMIZED algorithm instance!" << std::endl;
    }

    std::cout << "\n=== Test completed successfully! ===" << std::endl;
    return 0;
}
