#include "voxelization_factory.hpp"
#include "spatial_entities.hpp"
#include "mesh_loader.hpp"
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <Eigen/Core>

int main() {
    std::cout << "=== ForceFlow Voxelizer Example ===" << std::endl;
    std::cout << "High-performance mesh voxelization with CPU fallback for other shapes" << std::endl;

    // Create ForceFlow voxelizer
    auto forceflow_voxelizer = voxelization::VoxelizationFactory::createAlgorithm(
        voxelization::VoxelizationFactory::AlgorithmType::FORCEFLOW);

    // Initialize voxel grid
    int grid_x = 100, grid_y = 100, grid_z = 50;
    double resolution_xy = 0.1, resolution_z = 0.1;
    double origin_x = -5.0, origin_y = -5.0, origin_z = -2.5;

    forceflow_voxelizer->initialize(grid_x, grid_y, grid_z, resolution_xy, resolution_z,
        origin_x, origin_y, origin_z);

    std::cout << "Grid dimensions: " << grid_x << "x" << grid_y << "x" << grid_z << std::endl;
    std::cout << "Resolution: " << resolution_xy << "m (XY), " << resolution_z << "m (Z)" << std::endl;
    std::cout << "Origin: (" << origin_x << ", " << origin_y << ", " << origin_z << ")" << std::endl;

    // Create test entities
    std::vector<std::shared_ptr<voxelization::SpatialEntity>> entities;

    // 1. Mesh entity (will use ForceFlow GPU acceleration)
    std::cout << "\n1. Creating mesh entity..." << std::endl;
    std::vector<Eigen::Vector3d> vertices = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };
    std::vector<std::vector<int>> faces = {
        {0, 1, 2}, {0, 2, 3}, // bottom
        {4, 7, 6}, {4, 6, 5}, // top
        {0, 4, 5}, {0, 5, 1}, // front
        {2, 6, 7}, {2, 7, 3}, // back
        {0, 3, 7}, {0, 7, 4}, // left
        {1, 5, 6}, {1, 6, 2}  // right
    };
    auto mesh_entity = std::make_shared<voxelization::MeshEntity>(vertices, faces);
    entities.push_back(mesh_entity);

    // 2. Box entity (will fallback to CPU)
    std::cout << "2. Creating box entity..." << std::endl;
    auto box_entity = std::make_shared<voxelization::BoxEntity>(3.0, 2.0, 0.5, 2.0, 1.5, 1.0);
    entities.push_back(box_entity);

    // 3. Sphere entity (will fallback to CPU)
    std::cout << "3. Creating sphere entity..." << std::endl;
    auto sphere_entity = std::make_shared<voxelization::SphereEntity>(-2.0, -1.0, 0.8, 0.8);
    entities.push_back(sphere_entity);

    // 4. Cylinder entity (will fallback to CPU)
    std::cout << "4. Creating cylinder entity..." << std::endl;
    auto cylinder_entity = std::make_shared<voxelization::CylinderEntity>(0.0, 3.0, 1.0, 0.5, 2.0);
    entities.push_back(cylinder_entity);

    // 5. Ellipsoid entity (will fallback to CPU)
    std::cout << "5. Creating ellipsoid entity..." << std::endl;
    auto ellipsoid_entity = std::make_shared<voxelization::EllipsoidEntity>(-3.0, 2.0, 0.3, 0.6, 0.4, 0.3);
    entities.push_back(ellipsoid_entity);

    // 6. Cone entity (will fallback to CPU)
    std::cout << "6. Creating cone entity..." << std::endl;
    auto cone_entity = std::make_shared<voxelization::ConeEntity>(2.0, -2.0, 0.75, 0.4, 1.5);
    entities.push_back(cone_entity);

    // Perform voxelization
    std::cout << "\n=== Starting ForceFlow Voxelization ===" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    int total_marked_voxels = forceflow_voxelizer->voxelizeEntities(entities, 0.0, 255);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n=== Voxelization Results ===" << std::endl;
    std::cout << "Total marked voxels: " << total_marked_voxels << std::endl;
    std::cout << "Voxelization time: " << duration.count() << " ms" << std::endl;
    std::cout << "Voxelization rate: " << (total_marked_voxels * 1000.0 / duration.count()) << " voxels/ms" << std::endl;

    // Analyze results
    const unsigned char* voxel_grid = forceflow_voxelizer->getVoxelGrid();
    int occupied_voxels = 0;
    for (int i = 0; i < grid_x * grid_y * grid_z; ++i) {
        if (voxel_grid[i] > 0) {
            occupied_voxels++;
        }
    }

    std::cout << "Occupied voxels: " << occupied_voxels << std::endl;
    std::cout << "Occupancy rate: " << (occupied_voxels * 100.0 / (grid_x * grid_y * grid_z)) << "%" << std::endl;

    // Test individual entity voxelization
    std::cout << "\n=== Individual Entity Voxelization ===" << std::endl;

    // Test mesh (ForceFlow GPU)
    auto mesh_start = std::chrono::high_resolution_clock::now();
    int mesh_voxels = forceflow_voxelizer->voxelizeEntity(mesh_entity, 0.0, 255);
    auto mesh_end = std::chrono::high_resolution_clock::now();
    auto mesh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mesh_end - mesh_start);
    std::cout << "Mesh voxelization: " << mesh_voxels << " voxels in " << mesh_duration.count() << " ms" << std::endl;

    // Test box (CPU fallback)
    auto box_start = std::chrono::high_resolution_clock::now();
    int box_voxels = forceflow_voxelizer->voxelizeEntity(box_entity, 0.0, 255);
    auto box_end = std::chrono::high_resolution_clock::now();
    auto box_duration = std::chrono::duration_cast<std::chrono::milliseconds>(box_end - box_start);
    std::cout << "Box voxelization: " << box_voxels << " voxels in " << box_duration.count() << " ms" << std::endl;

    // Test sphere (CPU fallback)
    auto sphere_start = std::chrono::high_resolution_clock::now();
    int sphere_voxels = forceflow_voxelizer->voxelizeEntity(sphere_entity, 0.0, 255);
    auto sphere_end = std::chrono::high_resolution_clock::now();
    auto sphere_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sphere_end - sphere_start);
    std::cout << "Sphere voxelization: " << sphere_voxels << " voxels in " << sphere_duration.count() << " ms" << std::endl;

    // Save results
    std::cout << "\n=== Saving Results ===" << std::endl;
    if (forceflow_voxelizer->saveToFile("forceflow_voxelization_result.bin")) {
        std::cout << "Voxelization result saved to forceflow_voxelization_result.bin" << std::endl;
    }
    else {
        std::cout << "Failed to save voxelization result" << std::endl;
    }

    std::cout << "\n=== ForceFlow Example Complete ===" << std::endl;
    std::cout << "ForceFlow voxelizer successfully processed:" << std::endl;
    std::cout << "  - Mesh entities: GPU-accelerated with ForceFlow algorithm" << std::endl;
    std::cout << "  - Other entities: CPU fallback for stability" << std::endl;
    std::cout << "  - Total entities: " << entities.size() << std::endl;
    std::cout << "  - Total voxels: " << total_marked_voxels << std::endl;

    return 0;
}
