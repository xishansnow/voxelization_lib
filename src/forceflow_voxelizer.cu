#include "forceflow_voxelizer.hpp"
#include "cpu_voxelization.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cuda_runtime.h>

#ifdef ENABLE_CUDA

// ForceFlow-inspired CUDA kernels for mesh voxelization
__global__ void voxelizeMeshKernelForceFlow(
    unsigned char* voxel_grid,
    int grid_x, int grid_y, int grid_z,
    double resolution_xy, double resolution_z,
    double origin_x, double origin_y, double origin_z,
    float3* vertices, int num_vertices,
    int3* faces, int num_faces,
    unsigned char cost_value,
    bool solid_voxelization) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid_x || y >= grid_y || z >= grid_z) return;

    // Convert voxel coordinates to world coordinates
    double world_x = origin_x + (x + 0.5) * resolution_xy;
    double world_y = origin_y + (y + 0.5) * resolution_xy;
    double world_z = origin_z + (z + 0.5) * resolution_z;

    // Simple point-in-mesh test (ray casting)
    // This is a simplified version - ForceFlow uses more sophisticated algorithms
    bool inside = false;
    int intersections = 0;

    // Cast ray in +X direction
    for (int i = 0; i < num_faces; ++i) {
        int3 face = faces[i];
        float3 v0 = vertices[face.x];
        float3 v1 = vertices[face.y];
        float3 v2 = vertices[face.z];

        // Simple triangle-ray intersection test
        // This is a basic implementation - ForceFlow uses optimized algorithms
        double ray_origin[3] = { world_x, world_y, world_z };
        double ray_direction[3] = { 1.0, 0.0, 0.0 };

        // Möller–Trumbore intersection algorithm (simplified)
        double edge1[3] = { v1.x - v0.x, v1.y - v0.y, v1.z - v0.z };
        double edge2[3] = { v2.x - v0.x, v2.y - v0.y, v2.z - v0.z };
        double h[3] = { ray_direction[1] * edge2[2] - ray_direction[2] * edge2[1],
                       ray_direction[2] * edge2[0] - ray_direction[0] * edge2[2],
                       ray_direction[0] * edge2[1] - ray_direction[1] * edge2[0] };

        double a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];
        if (fabs(a) < 1e-8) continue; // Ray is parallel to triangle

        double f = 1.0 / a;
        double s[3] = { ray_origin[0] - v0.x, ray_origin[1] - v0.y, ray_origin[2] - v0.z };
        double u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);

        if (u < 0.0 || u > 1.0) continue;

        double q[3] = { s[1] * edge1[2] - s[2] * edge1[1],
                       s[2] * edge1[0] - s[0] * edge1[2],
                       s[0] * edge1[1] - s[1] * edge1[0] };
        double v = f * (ray_direction[0] * q[0] + ray_direction[1] * q[1] + ray_direction[2] * q[2]);

        if (v < 0.0 || u + v > 1.0) continue;

        double t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);

        if (t > 0.0) {
            intersections++;
        }
    }

    // Odd number of intersections means inside
    inside = (intersections % 2) == 1;

    if (inside) {
        int index = z * grid_x * grid_y + y * grid_x + x;
        voxel_grid[index] = cost_value;
    }
}

// Morton encoding for memory access optimization
__device__ __host__ unsigned int mortonEncode(unsigned int x, unsigned int y, unsigned int z) {
    unsigned int result = 0;
    for (int i = 0; i < 10; ++i) { // Assuming 10 bits per dimension (1024 max)
        result |= ((x & (1 << i)) << (2 * i)) |
            ((y & (1 << i)) << (2 * i + 1)) |
            ((z & (1 << i)) << (2 * i + 2));
    }
    return result;
}

#endif // ENABLE_CUDA

namespace voxelization {

    ForceFlowVoxelizer::ForceFlowVoxelizer()
        : grid_x_(0), grid_y_(0), grid_z_(0), resolution_xy_(0.0), resolution_z_(0.0),
        origin_x_(0.0), origin_y_(0.0), origin_z_(0.0), d_voxel_grid_(nullptr),
        grid_size_(0), gpu_available_(false), solid_voxelization_(false) {

#ifdef ENABLE_CUDA
        // Check CUDA availability
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error == cudaSuccess && deviceCount > 0) {
            gpu_available_ = true;
            std::cout << "ForceFlow GPU voxelizer: CUDA GPU detected. High-performance mesh voxelization enabled." << std::endl;
        }
        else {
            gpu_available_ = false;
            std::cout << "ForceFlow GPU voxelizer: CUDA GPU not available. Falling back to CPU implementation." << std::endl;
        }
#else
        gpu_available_ = false;
        std::cout << "ForceFlow GPU voxelizer: CUDA support not compiled. Using CPU implementation." << std::endl;
#endif
    }

    ForceFlowVoxelizer::~ForceFlowVoxelizer() {
        freeGPUMemory();
    }

    void ForceFlowVoxelizer::initialize(int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z) {
        grid_x_ = grid_x;
        grid_y_ = grid_y;
        grid_z_ = grid_z;
        resolution_xy_ = resolution_xy;
        resolution_z_ = resolution_z;
        origin_x_ = origin_x;
        origin_y_ = origin_y;
        origin_z_ = origin_z;

        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_, 0);
        grid_size_ = voxel_grid_.size();

        if (gpu_available_) {
            allocateGPUMemory();
        }
    }

    int ForceFlowVoxelizer::voxelizeEntity(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {

        // Check if it's a mesh entity
        if (auto mesh = std::dynamic_pointer_cast<MeshEntity>(entity)) {
            return voxelizeMeshForceFlow(mesh, buffer_size, cost_value);
        }
        else {
            // Fallback to CPU for non-mesh entities
            return fallbackToCPU(entity, buffer_size, cost_value);
        }
    }

    int ForceFlowVoxelizer::voxelizeEntities(const std::vector<std::shared_ptr<SpatialEntity>>& entities,
        double buffer_size, unsigned char cost_value) {

        int total_marked_voxels = 0;

        for (const auto& entity : entities) {
            total_marked_voxels += voxelizeEntity(entity, buffer_size, cost_value);
        }

        return total_marked_voxels;
    }

    int ForceFlowVoxelizer::voxelizeMeshForceFlow(const std::shared_ptr<MeshEntity>& mesh,
        double buffer_size, unsigned char cost_value) {

#ifdef ENABLE_CUDA
        if (!gpu_available_) {
            return fallbackToCPU(mesh, buffer_size, cost_value);
        }

        // Get mesh data
        const auto& vertices = mesh->getVertices();
        const auto& faces = mesh->getFaces();

        if (vertices.empty() || faces.empty()) {
            std::cerr << "ForceFlow: Empty mesh data" << std::endl;
            return -1;
        }

        // Prepare vertex data for GPU
        std::vector<float3> vertex_data;
        vertex_data.reserve(vertices.size());
        for (const auto& v : vertices) {
            vertex_data.push_back(make_float3(v.x(), v.y(), v.z()));
        }

        // Prepare face data for GPU
        std::vector<int3> face_data;
        face_data.reserve(faces.size());
        for (const auto& face : faces) {
            if (face.size() >= 3) {
                face_data.push_back(make_int3(face[0], face[1], face[2]));
            }
        }

        // Allocate GPU memory for mesh data
        float3* d_vertices;
        int3* d_faces;
        cudaMalloc(&d_vertices, vertex_data.size() * sizeof(float3));
        cudaMalloc(&d_faces, face_data.size() * sizeof(int3));

        // Copy data to GPU
        cudaMemcpy(d_vertices, vertex_data.data(), vertex_data.size() * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_faces, face_data.data(), face_data.size() * sizeof(int3), cudaMemcpyHostToDevice);

        // Define block and grid dimensions (optimized for ForceFlow)
        dim3 blockSize(16, 16, 4);  // 1024 threads per block
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);

        // Launch ForceFlow mesh voxelization kernel
        voxelizeMeshKernelForceFlow << <gridSize, blockSize >> > (
            d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            d_vertices, vertices.size(), d_faces, face_data.size(), cost_value, solid_voxelization_);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "ForceFlow CUDA error: " << cudaGetErrorString(error) << std::endl;
            cudaFree(d_vertices);
            cudaFree(d_faces);
            return -1;
        }

        // Synchronize and copy results back
        cudaDeviceSynchronize();
        copyFromGPU();

        // Free GPU memory
        cudaFree(d_vertices);
        cudaFree(d_faces);

        // Count marked voxels
        int marked_voxels = 0;
        for (unsigned char voxel : voxel_grid_) {
            if (voxel == cost_value) {
                marked_voxels++;
            }
        }

        return marked_voxels;
#else
        return fallbackToCPU(mesh, buffer_size, cost_value);
#endif
    }

    int ForceFlowVoxelizer::fallbackToCPU(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {

        // Use CPU sequential voxelizer as fallback
        CPUSequentialVoxelization cpu_voxelizer;
        cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);

        return cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
    }

    // Morton encoding utilities
    unsigned int ForceFlowVoxelizer::mortonEncode(unsigned int x, unsigned int y, unsigned int z) const {
        unsigned int result = 0;
        for (int i = 0; i < 10; ++i) {
            result |= ((x & (1 << i)) << (2 * i)) |
                ((y & (1 << i)) << (2 * i + 1)) |
                ((z & (1 << i)) << (2 * i + 2));
        }
        return result;
    }

    void ForceFlowVoxelizer::mortonDecode(unsigned int morton, unsigned int& x, unsigned int& y, unsigned int& z) const {
        x = y = z = 0;
        for (int i = 0; i < 10; ++i) {
            x |= (morton & (1 << (3 * i))) >> (2 * i);
            y |= (morton & (1 << (3 * i + 1))) >> (2 * i + 1);
            z |= (morton & (1 << (3 * i + 2))) >> (2 * i + 2);
        }
    }

    std::vector<double> ForceFlowVoxelizer::calculateMeshBoundingBox(const std::shared_ptr<MeshEntity>& mesh) const {
        const auto& vertices = mesh->getVertices();
        if (vertices.empty()) {
            return { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        }

        double min_x = vertices[0].x(), max_x = vertices[0].x();
        double min_y = vertices[0].y(), max_y = vertices[0].y();
        double min_z = vertices[0].z(), max_z = vertices[0].z();

        for (const auto& v : vertices) {
            min_x = std::min(min_x, v.x());
            max_x = std::max(max_x, v.x());
            min_y = std::min(min_y, v.y());
            max_y = std::max(max_y, v.y());
            min_z = std::min(min_z, v.z());
            max_z = std::max(max_z, v.z());
        }

        return { min_x, min_y, min_z, max_x, max_y, max_z };
    }

    // Standard VoxelizationBase interface implementations
    bool ForceFlowVoxelizer::worldToVoxel(double world_x, double world_y, double world_z,
        int& voxel_x, int& voxel_y, int& voxel_z) const {
        voxel_x = static_cast<int>((world_x - origin_x_) / resolution_xy_);
        voxel_y = static_cast<int>((world_y - origin_y_) / resolution_xy_);
        voxel_z = static_cast<int>((world_z - origin_z_) / resolution_z_);
        return isValidVoxel(voxel_x, voxel_y, voxel_z);
    }

    void ForceFlowVoxelizer::voxelToWorld(int voxel_x, int voxel_y, int voxel_z,
        double& world_x, double& world_y, double& world_z) const {
        world_x = origin_x_ + (voxel_x + 0.5) * resolution_xy_;
        world_y = origin_y_ + (voxel_y + 0.5) * resolution_xy_;
        world_z = origin_z_ + (voxel_z + 0.5) * resolution_z_;
    }

    bool ForceFlowVoxelizer::isValidVoxel(int voxel_x, int voxel_y, int voxel_z) const {
        return voxel_x >= 0 && voxel_x < grid_x_ &&
            voxel_y >= 0 && voxel_y < grid_y_ &&
            voxel_z >= 0 && voxel_z < grid_z_;
    }

    void ForceFlowVoxelizer::updateVoxelCost(int voxel_x, int voxel_y, int voxel_z, unsigned char cost_value) {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            voxel_grid_[index] = cost_value;
        }
    }

    unsigned char ForceFlowVoxelizer::getVoxelCost(int voxel_x, int voxel_y, int voxel_z) const {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            return voxel_grid_[index];
        }
        return 0;
    }

    void ForceFlowVoxelizer::clear() {
        std::fill(voxel_grid_.begin(), voxel_grid_.end(), 0);
        if (gpu_available_ && d_voxel_grid_ != nullptr) {
            cudaMemset(d_voxel_grid_, 0, grid_size_);
        }
    }

    void ForceFlowVoxelizer::allocateGPUMemory() {
#ifdef ENABLE_CUDA
        if (d_voxel_grid_ != nullptr) {
            cudaFree(d_voxel_grid_);
        }
        cudaMalloc(&d_voxel_grid_, grid_size_);
        cudaMemset(d_voxel_grid_, 0, grid_size_);
#endif
    }

    void ForceFlowVoxelizer::freeGPUMemory() {
#ifdef ENABLE_CUDA
        if (d_voxel_grid_ != nullptr) {
            cudaFree(d_voxel_grid_);
            d_voxel_grid_ = nullptr;
        }
#endif
    }

    void ForceFlowVoxelizer::copyToGPU() {
#ifdef ENABLE_CUDA
        if (gpu_available_ && d_voxel_grid_ != nullptr) {
            cudaMemcpy(d_voxel_grid_, voxel_grid_.data(), grid_size_, cudaMemcpyHostToDevice);
        }
#endif
    }

    void ForceFlowVoxelizer::copyFromGPU() {
#ifdef ENABLE_CUDA
        if (gpu_available_ && d_voxel_grid_ != nullptr) {
            cudaMemcpy(voxel_grid_.data(), d_voxel_grid_, grid_size_, cudaMemcpyDeviceToHost);
        }
#endif
    }

    void ForceFlowVoxelizer::setResolution(int x, int y, int z) {
        grid_x_ = x;
        grid_y_ = y;
        grid_z_ = z;
        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_, 0);
        grid_size_ = voxel_grid_.size();
        if (gpu_available_) {
            allocateGPUMemory();
        }
    }

    void ForceFlowVoxelizer::setBoundingBox(double min_x, double min_y, double min_z,
        double max_x, double max_y, double max_z) {
        origin_x_ = min_x;
        origin_y_ = min_y;
        origin_z_ = min_z;
        resolution_xy_ = (max_x - min_x) / grid_x_;
        resolution_z_ = (max_z - min_z) / grid_z_;
    }

    bool ForceFlowVoxelizer::voxelize(const std::vector<std::shared_ptr<SpatialEntity>>& entities) {
        clear();
        return voxelizeEntities(entities) > 0;
    }

    bool ForceFlowVoxelizer::saveToFile(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;
        file.write(reinterpret_cast<const char*>(&grid_x_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&grid_y_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&grid_z_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&resolution_xy_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&resolution_z_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&origin_x_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&origin_y_), sizeof(double));
        file.write(reinterpret_cast<const char*>(&origin_z_), sizeof(double));
        file.write(reinterpret_cast<const char*>(voxel_grid_.data()), voxel_grid_.size());
        file.close();
        return true;
    }

    bool ForceFlowVoxelizer::loadFromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;
        int grid_x, grid_y, grid_z;
        double resolution_xy, resolution_z, origin_x, origin_y, origin_z;
        file.read(reinterpret_cast<char*>(&grid_x), sizeof(int));
        file.read(reinterpret_cast<char*>(&grid_y), sizeof(int));
        file.read(reinterpret_cast<char*>(&grid_z), sizeof(int));
        file.read(reinterpret_cast<char*>(&resolution_xy), sizeof(double));
        file.read(reinterpret_cast<char*>(&resolution_z), sizeof(double));
        file.read(reinterpret_cast<char*>(&origin_x), sizeof(double));
        file.read(reinterpret_cast<char*>(&origin_y), sizeof(double));
        file.read(reinterpret_cast<char*>(&origin_z), sizeof(double));
        initialize(grid_x, grid_y, grid_z, resolution_xy, resolution_z, origin_x, origin_y, origin_z);
        file.read(reinterpret_cast<char*>(voxel_grid_.data()), voxel_grid_.size());
        file.close();
        return true;
    }

    bool ForceFlowVoxelizer::saveVoxelGrid(const std::string& filename) const {
        return saveToFile(filename);
    }

    void ForceFlowVoxelizer::getVoxelGrid(std::vector<unsigned char>& out) const {
        out = voxel_grid_;
    }

} // namespace voxelization
