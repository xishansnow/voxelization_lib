#include "voxelization_factory.hpp"
#include "spatial_entities.hpp"
#include <iostream>
#include <chrono>
#include <memory>

using namespace voxelization;

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
    
    std::vector<std::shared_ptr<SpatialEntity>> entities = {box_entity, sphere_entity, cylinder_entity};
    
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
    } else {
        std::cout << "   - ERROR: Failed to save voxel grid" << std::endl;
    }
    
    // Test 10: Available algorithms
    std::cout << "\n10. Available algorithms:" << std::endl;
    auto algorithms = VoxelizationFactory::getAvailableAlgorithms();
    for (const auto& alg : algorithms) {
        std::cout << "   - " << alg << std::endl;
    }
    
    std::cout << "\n=== Test completed successfully! ===" << std::endl;
    return 0;
} 