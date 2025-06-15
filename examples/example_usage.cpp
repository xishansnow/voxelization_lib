#include "voxelization_factory.hpp"
#include "spatial_entities.hpp"
#include <iostream>
#include <memory>

using namespace voxelization;

int main() {
    std::cout << "=== Voxelization Library Usage Example ===" << std::endl;
    
    // Step 1: Create spatial entities
    std::cout << "\n1. Creating spatial entities..." << std::endl;
    
    // Create a box obstacle
    auto box = std::make_shared<BoxEntity>(10.0, 10.0, 2.0, 4.0, 4.0, 4.0);
    
    // Create a cylindrical obstacle
    auto cylinder = std::make_shared<CylinderEntity>(5.0, 15.0, 1.5, 2.0, 3.0);
    
    // Create a spherical obstacle
    auto sphere = std::make_shared<SphereEntity>(15.0, 5.0, 1.0, 1.5);
    
    // Create a composite entity (union of multiple entities)
    std::vector<std::shared_ptr<SpatialEntity>> composite_entities = {box, cylinder, sphere};
    auto composite = std::make_shared<CompositeEntity>(composite_entities);
    
    std::cout << "   - Created box, cylinder, sphere, and composite entities" << std::endl;
    
    // Step 2: Create voxelization algorithm
    std::cout << "\n2. Creating voxelization algorithm..." << std::endl;
    
    auto voxelizer = VoxelizationFactory::createAlgorithm(VoxelizationFactory::AlgorithmType::CPU_PARALLEL);
    if (!voxelizer) {
        std::cout << "   ERROR: Failed to create voxelization algorithm!" << std::endl;
        return -1;
    }
    
    std::cout << "   - Created CPU Parallel voxelization algorithm" << std::endl;
    
    // Step 3: Initialize voxelization grid
    std::cout << "\n3. Initializing voxelization grid..." << std::endl;
    
    // Define grid parameters
    int grid_x = 200, grid_y = 200, grid_z = 100;
    double resolution_xy = 0.1;  // 10cm resolution in XY
    double resolution_z = 0.1;   // 10cm resolution in Z
    double origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;
    
    voxelizer->initialize(grid_x, grid_y, grid_z, resolution_xy, resolution_z, origin_x, origin_y, origin_z);
    
    std::cout << "   - Grid size: " << grid_x << "x" << grid_y << "x" << grid_z << std::endl;
    std::cout << "   - Resolution: " << resolution_xy << "m (xy), " << resolution_z << "m (z)" << std::endl;
    std::cout << "   - World bounds: [" << origin_x << ", " << origin_y << ", " << origin_z << "] to ["
              << origin_x + grid_x * resolution_xy << ", " 
              << origin_y + grid_y * resolution_xy << ", " 
              << origin_z + grid_z * resolution_z << "]" << std::endl;
    
    // Step 4: Voxelize individual entities
    std::cout << "\n4. Voxelizing individual entities..." << std::endl;
    
    int box_voxels = voxelizer->voxelizeEntity(box, 0.2, 255);  // 20cm buffer
    std::cout << "   - Box voxelized: " << box_voxels << " voxels marked" << std::endl;
    
    int cylinder_voxels = voxelizer->voxelizeEntity(cylinder, 0.1, 200);  // 10cm buffer
    std::cout << "   - Cylinder voxelized: " << cylinder_voxels << " voxels marked" << std::endl;
    
    int sphere_voxels = voxelizer->voxelizeEntity(sphere, 0.0, 150);  // No buffer
    std::cout << "   - Sphere voxelized: " << sphere_voxels << " voxels marked" << std::endl;
    
    // Step 5: Test coordinate conversion
    std::cout << "\n5. Testing coordinate conversion..." << std::endl;
    
    // Convert world coordinates to voxel coordinates
    int voxel_x, voxel_y, voxel_z;
    if (voxelizer->worldToVoxel(10.0, 10.0, 2.0, voxel_x, voxel_y, voxel_z)) {
        std::cout << "   - World (10, 10, 2) -> Voxel (" << voxel_x << ", " << voxel_y << ", " << voxel_z << ")" << std::endl;
        
        // Check if this voxel is occupied
        unsigned char cost = voxelizer->getVoxelCost(voxel_x, voxel_y, voxel_z);
        std::cout << "   - Voxel cost at this location: " << static_cast<int>(cost) << std::endl;
    }
    
    // Convert voxel coordinates to world coordinates
    double world_x, world_y, world_z;
    voxelizer->voxelToWorld(100, 100, 20, world_x, world_y, world_z);
    std::cout << "   - Voxel (100, 100, 20) -> World (" << world_x << ", " << world_y << ", " << world_z << ")" << std::endl;
    
    // Step 6: Clear and voxelize composite entity
    std::cout << "\n6. Voxelizing composite entity..." << std::endl;
    
    voxelizer->clear();
    int composite_voxels = voxelizer->voxelizeEntity(composite, 0.1, 255);
    std::cout << "   - Composite entity voxelized: " << composite_voxels << " voxels marked" << std::endl;
    
    // Step 7: Save voxel grid to file
    std::cout << "\n7. Saving voxel grid..." << std::endl;
    
    if (voxelizer->saveToFile("example_voxel_grid.bin")) {
        std::cout << "   - Voxel grid saved to example_voxel_grid.bin" << std::endl;
    } else {
        std::cout << "   - ERROR: Failed to save voxel grid" << std::endl;
    }
    
    // Step 8: Demonstrate different algorithms
    std::cout << "\n8. Comparing different algorithms..." << std::endl;
    
    auto algorithms = VoxelizationFactory::getAvailableAlgorithms();
    for (const auto& alg_name : algorithms) {
        auto alg = VoxelizationFactory::createAlgorithm(
            VoxelizationFactory::getAlgorithmType(alg_name));
        
        if (alg) {
            alg->initialize(grid_x, grid_y, grid_z, resolution_xy, resolution_z, origin_x, origin_y, origin_z);
            int voxels = alg->voxelizeEntity(box, 0.0, 255);
            std::cout << "   - " << alg_name << ": " << voxels << " voxels marked" << std::endl;
        }
    }
    
    // Step 9: Demonstrate entity properties
    std::cout << "\n9. Entity properties..." << std::endl;
    
    auto box_props = box->getProperties();
    std::cout << "   - Box properties:" << std::endl;
    for (const auto& prop : box_props) {
        std::cout << "     " << prop.first << ": " << prop.second << std::endl;
    }
    
    auto sphere_props = sphere->getProperties();
    std::cout << "   - Sphere properties:" << std::endl;
    for (const auto& prop : sphere_props) {
        std::cout << "     " << prop.first << ": " << prop.second << std::endl;
    }
    
    // Step 10: Demonstrate point-in-entity testing
    std::cout << "\n10. Point-in-entity testing..." << std::endl;
    
    std::vector<std::pair<std::string, std::shared_ptr<SpatialEntity>>> test_entities = {
        {"Box", box},
        {"Cylinder", cylinder},
        {"Sphere", sphere}
    };
    
    std::vector<std::tuple<double, double, double, std::string>> test_points = {
        {10.0, 10.0, 2.0, "Box center"},
        {5.0, 15.0, 1.5, "Cylinder center"},
        {15.0, 5.0, 1.0, "Sphere center"},
        {0.0, 0.0, 0.0, "Origin"},
        {20.0, 20.0, 10.0, "Far point"}
    };
    
    for (const auto& entity_pair : test_entities) {
        const std::string& entity_name = entity_pair.first;
        const std::shared_ptr<SpatialEntity>& entity = entity_pair.second;
        std::cout << "   - " << entity_name << ":" << std::endl;
        for (const auto& point_tuple : test_points) {
            double x = std::get<0>(point_tuple);
            double y = std::get<1>(point_tuple);
            double z = std::get<2>(point_tuple);
            std::string desc = std::get<3>(point_tuple);
            bool inside = entity->isPointInside(x, y, z);
            std::cout << "     " << desc << " (" << x << ", " << y << ", " << z << "): " 
                      << (inside ? "INSIDE" : "OUTSIDE") << std::endl;
        }
    }
    
    std::cout << "\n=== Example completed successfully! ===" << std::endl;
    std::cout << "Generated files:" << std::endl;
    std::cout << "  - example_voxel_grid.bin: Binary voxel grid file" << std::endl;
    
    return 0;
} 