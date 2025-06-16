#include "voxelization_algorithms.hpp"
#include "mesh_loader.hpp"
#include "voxelization_factory.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <3d_model_file> [voxel_resolution]" << std::endl;
        std::cout << "Supported formats: OBJ, FBX, 3DS, DAE, PLY, STL, etc." << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int resolution = (argc > 2) ? std::stoi(argv[2]) : 64;

    std::cout << "=== 3D Mesh Voxelization Example ===" << std::endl;
    std::cout << "Loading file: " << filename << std::endl;
    std::cout << "Voxel resolution: " << resolution << "x" << resolution << "x" << resolution << std::endl;

    // Create mesh loader
    voxelization::MeshLoader loader;

    // Check if format is supported
    std::string extension = filename.substr(filename.find_last_of('.'));
    if (!loader.isFormatSupported(extension)) {
        std::cout << "Warning: File format " << extension << " might not be supported." << std::endl;
        std::cout << "Supported formats:" << std::endl;
        for (const auto& fmt : loader.getSupportedFormats()) {
            std::cout << "  " << fmt << std::endl;
        }
    }

    // Load the mesh
    auto mesh_entity = loader.loadMesh(filename);
    if (!mesh_entity) {
        std::cerr << "Failed to load mesh: " << loader.getLastError() << std::endl;
        return 1;
    }

    if (!mesh_entity->isValid()) {
        std::cerr << "Loaded mesh is invalid or empty" << std::endl;
        return 1;
    }

    // Print mesh information
    const auto& vertices = mesh_entity->getVertices();
    const auto& faces = mesh_entity->getFaces();
    std::cout << "\nMesh Information:" << std::endl;
    std::cout << "  Vertices: " << vertices.size() << std::endl;
    std::cout << "  Triangles: " << faces.size() / 3 << std::endl;
    // std::cout << "  Normals: " << mesh_entity->getNormals().size() << std::endl;
    // std::cout << "  Texture coordinates: " << mesh_entity->getTexCoords().size() << std::endl;

    auto bbox = mesh_entity->getBoundingBox();
    std::cout << "  Bounding box: ("
        << bbox[0] << ", " << bbox[1] << ", " << bbox[2] << ") to ("
        << bbox[3] << ", " << bbox[4] << ", " << bbox[5] << ")" << std::endl;

    auto props = mesh_entity->getProperties();
    std::cout << "  Volume: " << props["volume"] << std::endl;

    // Create voxelization algorithm
    std::cout << "\nCreating voxelization algorithm..." << std::endl;
    auto voxelizer = voxelization::VoxelizationFactory::createAlgorithm(
        voxelization::VoxelizationFactory::AlgorithmType::CPU_PARALLEL
    );

    if (!voxelizer) {
        std::cerr << "Failed to create voxelization algorithm" << std::endl;
        return 1;
    }

    // Set voxelization parameters
    voxelizer->setResolution(resolution, resolution, resolution);
    voxelizer->setBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);

    // Add the mesh entity
    std::vector<std::shared_ptr<voxelization::SpatialEntity>> entities;
    entities.push_back(mesh_entity);

    // Perform voxelization
    std::cout << "Performing voxelization..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    bool success = voxelizer->voxelize(entities);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (!success) {
        std::cerr << "Voxelization failed!" << std::endl;
        return 1;
    }

    std::cout << "Voxelization completed in " << duration.count() << " ms" << std::endl;

    // Get voxel grid
    const unsigned char* voxel_grid = voxelizer->getVoxelGrid();
    if (!voxel_grid) {
        std::cerr << "Failed to get voxel grid" << std::endl;
        return 1;
    }

    // Analyze voxel grid
    int occupied_voxels = 0;
    int total_voxels = resolution * resolution * resolution;

    for (int i = 0; i < total_voxels; ++i) {
        if (voxel_grid[i] > 0) {
            occupied_voxels++;
        }
    }

    double occupancy_rate = static_cast<double>(occupied_voxels) / total_voxels * 100.0;

    std::cout << "\nVoxelization Results:" << std::endl;
    std::cout << "  Total voxels: " << total_voxels << std::endl;
    std::cout << "  Occupied voxels: " << occupied_voxels << std::endl;
    std::cout << "  Occupancy rate: " << occupancy_rate << "%" << std::endl;

    // Save voxel grid to file
    std::string output_filename = filename.substr(0, filename.find_last_of('.')) + "_voxels.raw";
    if (voxelizer->saveVoxelGrid(output_filename)) {
        std::cout << "Voxel grid saved to: " << output_filename << std::endl;
    }
    else {
        std::cout << "Failed to save voxel grid" << std::endl;
    }

    // Test point-in-mesh functionality
    std::cout << "\nTesting point-in-mesh functionality..." << std::endl;

    // Test center point
    double center_x = (bbox[0] + bbox[3]) / 2.0;
    double center_y = (bbox[1] + bbox[4]) / 2.0;
    double center_z = (bbox[2] + bbox[5]) / 2.0;

    bool center_inside = mesh_entity->isPointInside(center_x, center_y, center_z);
    std::cout << "  Center point (" << center_x << ", " << center_y << ", " << center_z
        << ") is " << (center_inside ? "inside" : "outside") << " the mesh" << std::endl;

    // Test corner point
    bool corner_inside = mesh_entity->isPointInside(bbox[0] - 1.0, bbox[1] - 1.0, bbox[2] - 1.0);
    std::cout << "  Corner point (" << (bbox[0] - 1.0) << ", " << (bbox[1] - 1.0) << ", " << (bbox[2] - 1.0)
        << ") is " << (corner_inside ? "inside" : "outside") << " the mesh" << std::endl;

    std::cout << "\n=== Voxelization Complete ===" << std::endl;
    return 0;
}
