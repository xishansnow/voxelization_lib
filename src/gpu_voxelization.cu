#include "gpu_voxelization.hpp"
#include "cpu_voxelization.hpp"
#include "spatial_entities.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

// CUDA includes with conditional compilation
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <cmath>
#endif

namespace voxelization {

#ifdef ENABLE_CUDA
    // CUDA kernel for voxelization
    __global__ void voxelizeBoxKernel(unsigned char* voxel_grid, int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z,
        double center_x, double center_y, double center_z,
        double size_x, double size_y, double size_z,
        unsigned char cost_value) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= grid_x || y >= grid_y || z >= grid_z) return;

        // Convert voxel coordinates to world coordinates
        double world_x = origin_x + (x + 0.5) * resolution_xy;
        double world_y = origin_y + (y + 0.5) * resolution_xy;
        double world_z = origin_z + (z + 0.5) * resolution_z;

        // Check if point is inside box
        if (world_x >= center_x - size_x / 2 && world_x <= center_x + size_x / 2 &&
            world_y >= center_y - size_y / 2 && world_y <= center_y + size_y / 2 &&
            world_z >= center_z - size_z / 2 && world_z <= center_z + size_z / 2) {

            int index = z * grid_x * grid_y + y * grid_x + x;
            voxel_grid[index] = cost_value;
        }
    }

    __global__ void voxelizeSphereKernel(unsigned char* voxel_grid, int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z,
        double center_x, double center_y, double center_z,
        double radius, unsigned char cost_value) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= grid_x || y >= grid_y || z >= grid_z) return;

        // Convert voxel coordinates to world coordinates
        double world_x = origin_x + (x + 0.5) * resolution_xy;
        double world_y = origin_y + (y + 0.5) * resolution_xy;
        double world_z = origin_z + (z + 0.5) * resolution_z;

        // Check if point is inside sphere
        double dx = world_x - center_x;
        double dy = world_y - center_y;
        double dz = world_z - center_z;
        double distance_squared = dx * dx + dy * dy + dz * dz;

        if (distance_squared <= radius * radius) {
            int index = z * grid_x * grid_y + y * grid_x + x;
            voxel_grid[index] = cost_value;
        }
    }

    __global__ void voxelizeCylinderKernel(unsigned char* voxel_grid, int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z,
        double center_x, double center_y, double center_z,
        double radius, double height, unsigned char cost_value) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= grid_x || y >= grid_y || z >= grid_z) return;

        // Convert voxel coordinates to world coordinates
        double world_x = origin_x + (x + 0.5) * resolution_xy;
        double world_y = origin_y + (y + 0.5) * resolution_xy;
        double world_z = origin_z + (z + 0.5) * resolution_z;

        // Check if point is inside cylinder
        double dx = world_x - center_x;
        double dy = world_y - center_y;
        double distance_squared = dx * dx + dy * dy;

        if (distance_squared <= radius * radius &&
            world_z >= center_z - height / 2 && world_z <= center_z + height / 2) {
            int index = z * grid_x * grid_y + y * grid_x + x;
            voxel_grid[index] = cost_value;
        }
    }

    // GPU kernel for mesh voxelization using ray casting
    __global__ void voxelizeMeshKernel(unsigned char* voxel_grid, int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z,
        double* vertices, int num_vertices,
        int* faces, int num_faces,
        unsigned char cost_value) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= grid_x || y >= grid_y || z >= grid_z) return;

        // Convert voxel coordinates to world coordinates
        double world_x = origin_x + (x + 0.5) * resolution_xy;
        double world_y = origin_y + (y + 0.5) * resolution_xy;
        double world_z = origin_z + (z + 0.5) * resolution_z;

        // Ray casting algorithm: shoot ray in +Z direction and count intersections
        int intersections = 0;
        double ray_origin[3] = { world_x, world_y, world_z };
        double ray_direction[3] = { 0.0, 0.0, 1.0 };

        // Check intersection with each triangle
        for (int i = 0; i < num_faces; i++) {
            int face_start = i * 3;
            int v1_idx = faces[face_start] * 3;
            int v2_idx = faces[face_start + 1] * 3;
            int v3_idx = faces[face_start + 2] * 3;

            // Triangle vertices
            double v1[3] = { vertices[v1_idx], vertices[v1_idx + 1], vertices[v1_idx + 2] };
            double v2[3] = { vertices[v2_idx], vertices[v2_idx + 1], vertices[v2_idx + 2] };
            double v3[3] = { vertices[v3_idx], vertices[v3_idx + 1], vertices[v3_idx + 2] };

            // Ray-triangle intersection test
            double edge1[3] = { v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2] };
            double edge2[3] = { v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2] };
            double h[3] = { ray_direction[1] * edge2[2] - ray_direction[2] * edge2[1],
                          ray_direction[2] * edge2[0] - ray_direction[0] * edge2[2],
                          ray_direction[0] * edge2[1] - ray_direction[1] * edge2[0] };

            double a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];
            if (fabs(a) < 1e-8) continue; // Ray is parallel to triangle

            double f = 1.0 / a;
            double s[3] = { ray_origin[0] - v1[0], ray_origin[1] - v1[1], ray_origin[2] - v1[2] };
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

        // If odd number of intersections, point is inside mesh
        if (intersections % 2 == 1) {
            int index = z * grid_x * grid_y + y * grid_x + x;
            voxel_grid[index] = cost_value;
        }
    }

    __global__ void voxelizeEllipsoidKernel(unsigned char* voxel_grid, int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z,
        double center_x, double center_y, double center_z,
        double radius_x, double radius_y, double radius_z,
        unsigned char cost_value) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x >= grid_x || y >= grid_y || z >= grid_z) return;
        double world_x = origin_x + (x + 0.5) * resolution_xy;
        double world_y = origin_y + (y + 0.5) * resolution_xy;
        double world_z = origin_z + (z + 0.5) * resolution_z;
        double dx = (world_x - center_x) / radius_x;
        double dy = (world_y - center_y) / radius_y;
        double dz = (world_z - center_z) / radius_z;
        double val = dx * dx + dy * dy + dz * dz;
        if (val <= 1.0) {
            int index = z * grid_x * grid_y + y * grid_x + x;
            voxel_grid[index] = cost_value;
    }
}

    __global__ void voxelizeConeKernel(unsigned char* voxel_grid, int grid_x, int grid_y, int grid_z,
        double resolution_xy, double resolution_z,
        double origin_x, double origin_y, double origin_z,
        double center_x, double center_y, double center_z,
        double radius, double height, unsigned char cost_value) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x >= grid_x || y >= grid_y || z >= grid_z) return;
        double world_x = origin_x + (x + 0.5) * resolution_xy;
        double world_y = origin_y + (y + 0.5) * resolution_xy;
        double world_z = origin_z + (z + 0.5) * resolution_z;
        double dz = world_z - center_z;
        if (dz >= 0 && dz <= height) {
            double r = radius * (1.0 - dz / height);
            double dx = world_x - center_x;
            double dy = world_y - center_y;
            double dist2 = dx * dx + dy * dy;
            if (dist2 <= r * r) {
                int index = z * grid_x * grid_y + y * grid_x + x;
                voxel_grid[index] = cost_value;
            }
        }
    }
#endif

    // GPUCudaVoxelization implementation
    GPUCudaVoxelization::GPUCudaVoxelization()
        : grid_x_(0), grid_y_(0), grid_z_(0), resolution_xy_(0.0), resolution_z_(0.0),
        origin_x_(0.0), origin_y_(0.0), origin_z_(0.0), d_voxel_grid_(nullptr),
        grid_size_(0), gpu_available_(false) {

#ifdef ENABLE_CUDA
        // Check CUDA availability
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error == cudaSuccess && deviceCount > 0) {
            gpu_available_ = true;
            std::cout << "CUDA GPU detected. GPU acceleration enabled." << std::endl;
        }
        else {
            gpu_available_ = false;
            std::cout << "CUDA GPU not available. Falling back to CPU implementation." << std::endl;
        }
#else
        gpu_available_ = false;
        std::cout << "CUDA support not compiled. Using CPU implementation." << std::endl;
#endif
    }

    GPUCudaVoxelization::~GPUCudaVoxelization() {
        freeGPUMemory();
    }

    void GPUCudaVoxelization::initialize(int grid_x, int grid_y, int grid_z,
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

    int GPUCudaVoxelization::voxelizeEntity(const std::shared_ptr<SpatialEntity>& entity,
        double buffer_size, unsigned char cost_value) {

        if (!gpu_available_) {
            // Fallback to CPU implementation
            CPUSequentialVoxelization cpu_voxelizer;
            cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);
            return cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
        }

        // GPU implementation
        if (auto box = std::dynamic_pointer_cast<BoxEntity>(entity)) {
            return voxelizeBox(box, buffer_size, cost_value);
        }
        else if (auto sphere = std::dynamic_pointer_cast<SphereEntity>(entity)) {
            return voxelizeSphere(sphere, buffer_size, cost_value);
        }
        else if (auto cylinder = std::dynamic_pointer_cast<CylinderEntity>(entity)) {
            return voxelizeCylinder(cylinder, buffer_size, cost_value);
        }
        else if (auto ellipsoid = std::dynamic_pointer_cast<EllipsoidEntity>(entity)) {
            return voxelizeEllipsoid(ellipsoid, buffer_size, cost_value);
        }
        else if (auto cone = std::dynamic_pointer_cast<ConeEntity>(entity)) {
            return voxelizeCone(cone, buffer_size, cost_value);
        }
        else if (auto mesh = std::dynamic_pointer_cast<MeshEntity>(entity)) {
            return voxelizeMesh(mesh, buffer_size, cost_value);
        }
        else {
            // Fallback to CPU implementation for unsupported types
            CPUSequentialVoxelization cpu_voxelizer;
            cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);
            return cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
        }
    }

    int GPUCudaVoxelization::voxelizeEntities(const std::vector<std::shared_ptr<SpatialEntity>>& entities,
        double buffer_size, unsigned char cost_value) {

        if (!gpu_available_) {
            // Fallback to CPU implementation
            CPUSequentialVoxelization cpu_voxelizer;
            cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);
            return cpu_voxelizer.voxelizeEntities(entities, buffer_size, cost_value);
        }

        // Optimized GPU implementation: batch all entities before GPU->CPU transfer
        int total_marked_voxels = 0;

        // Process all entities on GPU first
        for (const auto& entity : entities) {
            if (auto box = std::dynamic_pointer_cast<BoxEntity>(entity)) {
                voxelizeBoxGPUOnly(box, buffer_size, cost_value);
            }
            else if (auto sphere = std::dynamic_pointer_cast<SphereEntity>(entity)) {
                voxelizeSphereGPUOnly(sphere, buffer_size, cost_value);
        }
            else if (auto cylinder = std::dynamic_pointer_cast<CylinderEntity>(entity)) {
                voxelizeCylinderGPUOnly(cylinder, buffer_size, cost_value);
            }
            else if (auto ellipsoid = std::dynamic_pointer_cast<EllipsoidEntity>(entity)) {
                voxelizeEllipsoidGPUOnly(ellipsoid, buffer_size, cost_value);
            }
            else if (auto cone = std::dynamic_pointer_cast<ConeEntity>(entity)) {
                voxelizeConeGPUOnly(cone, buffer_size, cost_value);
            }
            else if (auto mesh = std::dynamic_pointer_cast<MeshEntity>(entity)) {
                voxelizeMeshGPUOnly(mesh, buffer_size, cost_value);
            }
            else {
                // Fallback to CPU implementation for unsupported types
                CPUSequentialVoxelization cpu_voxelizer;
                cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                    origin_x_, origin_y_, origin_z_);
                total_marked_voxels += cpu_voxelizer.voxelizeEntity(entity, buffer_size, cost_value);
            }
        }

        // Single GPU->CPU transfer after all entities are processed
        cudaDeviceSynchronize();
        copyFromGPU();

        // Count marked voxels once
        for (unsigned char voxel : voxel_grid_) {
            if (voxel == cost_value) {
                total_marked_voxels++;
            }
        }

        return total_marked_voxels;
            }

    int GPUCudaVoxelization::voxelizeBox(const std::shared_ptr<BoxEntity>& box, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        double center_x = box->getProperties()["center_x"];
        double center_y = box->getProperties()["center_y"];
        double center_z = box->getProperties()["center_z"];
        double size_x = box->getProperties()["size_x"] + 2 * buffer_size;
        double size_y = box->getProperties()["size_y"] + 2 * buffer_size;
        double size_z = box->getProperties()["size_z"] + 2 * buffer_size;
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);
        voxelizeBoxKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, size_x, size_y, size_z, cost_value);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return -1;
        }
        cudaDeviceSynchronize();
        copyFromGPU();

        // Count marked voxels
        int marked_voxels = 0;
        for (unsigned char voxel : voxel_grid_) {
            if (voxel == cost_value) {
                marked_voxels++;
            }
        }
        return marked_voxels;
#else
        return 0;
#endif
    }

    void GPUCudaVoxelization::voxelizeBoxGPUOnly(const std::shared_ptr<BoxEntity>& box, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        double center_x = box->getProperties()["center_x"];
        double center_y = box->getProperties()["center_y"];
        double center_z = box->getProperties()["center_z"];
        double size_x = box->getProperties()["size_x"] + 2 * buffer_size;
        double size_y = box->getProperties()["size_y"] + 2 * buffer_size;
        double size_z = box->getProperties()["size_z"] + 2 * buffer_size;
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);
        voxelizeBoxKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, size_x, size_y, size_z, cost_value);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }
#endif
    }

    int GPUCudaVoxelization::voxelizeSphere(const std::shared_ptr<SphereEntity>& sphere, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        double center_x = sphere->getProperties()["center_x"];
        double center_y = sphere->getProperties()["center_y"];
        double center_z = sphere->getProperties()["center_z"];
        double radius = sphere->getProperties()["radius"] + buffer_size;
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);
        voxelizeSphereKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, radius, cost_value);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return -1;
        }
        cudaDeviceSynchronize();
        copyFromGPU();

        // Count marked voxels
        int marked_voxels = 0;
        for (unsigned char voxel : voxel_grid_) {
            if (voxel == cost_value) {
                marked_voxels++;
            }
        }
        return marked_voxels;
#else
        return 0;
#endif
    }

    void GPUCudaVoxelization::voxelizeSphereGPUOnly(const std::shared_ptr<SphereEntity>& sphere, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        double center_x = sphere->getProperties()["center_x"];
        double center_y = sphere->getProperties()["center_y"];
        double center_z = sphere->getProperties()["center_z"];
        double radius = sphere->getProperties()["radius"] + buffer_size;
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);
        voxelizeSphereKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, radius, cost_value);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }
#endif
    }

    int GPUCudaVoxelization::voxelizeCylinder(const std::shared_ptr<CylinderEntity>& cylinder, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        double center_x = cylinder->getProperties()["center_x"];
        double center_y = cylinder->getProperties()["center_y"];
        double center_z = cylinder->getProperties()["center_z"];
        double radius = cylinder->getProperties()["radius"] + buffer_size;
        double height = cylinder->getProperties()["height"] + 2 * buffer_size;
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);
        voxelizeCylinderKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, radius, height, cost_value);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return -1;
        }
        cudaDeviceSynchronize();
        copyFromGPU();

        // Count marked voxels
        int marked_voxels = 0;
        for (unsigned char voxel : voxel_grid_) {
            if (voxel == cost_value) {
                marked_voxels++;
            }
        }
        return marked_voxels;
#else
        return 0;
#endif
    }

    void GPUCudaVoxelization::voxelizeCylinderGPUOnly(const std::shared_ptr<CylinderEntity>& cylinder, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        double center_x = cylinder->getProperties()["center_x"];
        double center_y = cylinder->getProperties()["center_y"];
        double center_z = cylinder->getProperties()["center_z"];
        double radius = cylinder->getProperties()["radius"] + buffer_size;
        double height = cylinder->getProperties()["height"] + 2 * buffer_size;
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);
        voxelizeCylinderKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, radius, height, cost_value);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }
#endif
    }

    int GPUCudaVoxelization::voxelizeMesh(const std::shared_ptr<MeshEntity>& mesh, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        if (!gpu_available_) {
            // Fallback to CPU implementation
            CPUSequentialVoxelization cpu_voxelizer;
            cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
                origin_x_, origin_y_, origin_z_);
            return cpu_voxelizer.voxelizeEntity(mesh, buffer_size, cost_value);
        }

        // Get mesh data
        const auto& vertices = mesh->getVertices();
        const auto& faces = mesh->getFaces();

        if (vertices.empty() || faces.empty()) {
            std::cerr << "Empty mesh data" << std::endl;
            return -1;
        }

        // Prepare vertex data for GPU
        std::vector<double> vertex_data;
        vertex_data.reserve(vertices.size() * 3);
        for (const auto& v : vertices) {
            vertex_data.push_back(v.x());
            vertex_data.push_back(v.y());
            vertex_data.push_back(v.z());
        }

        // Prepare face data for GPU (assuming triangular faces)
        std::vector<int> face_data;
        face_data.reserve(faces.size() * 3);
        for (const auto& face : faces) {
            if (face.size() >= 3) {
                face_data.push_back(face[0]);
                face_data.push_back(face[1]);
                face_data.push_back(face[2]);
            }
        }

        // Allocate GPU memory for mesh data
        double* d_vertices;
        int* d_faces;
        cudaMalloc(&d_vertices, vertex_data.size() * sizeof(double));
        cudaMalloc(&d_faces, face_data.size() * sizeof(int));

        // Copy data to GPU
        cudaMemcpy(d_vertices, vertex_data.data(), vertex_data.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_faces, face_data.data(), face_data.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Define block and grid dimensions
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);

        // Launch mesh voxelization kernel
        voxelizeMeshKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            d_vertices, vertices.size(), d_faces, face_data.size() / 3, cost_value);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
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

        return 0;
#else
        // Fallback to CPU implementation
        CPUSequentialVoxelization cpu_voxelizer;
        cpu_voxelizer.initialize(grid_x_, grid_y_, grid_z_, resolution_xy_, resolution_z_,
            origin_x_, origin_y_, origin_z_);
        return cpu_voxelizer.voxelizeEntity(mesh, buffer_size, cost_value);
#endif
        }

    void GPUCudaVoxelization::voxelizeMeshGPUOnly(const std::shared_ptr<MeshEntity>& mesh, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        // Get mesh data
        const auto& vertices = mesh->getVertices();
        const auto& faces = mesh->getFaces();

        if (vertices.empty() || faces.empty()) {
            std::cerr << "Empty mesh data" << std::endl;
            return;
        }

        // Prepare vertex data for GPU
        std::vector<double> vertex_data;
        vertex_data.reserve(vertices.size() * 3);
        for (const auto& v : vertices) {
            vertex_data.push_back(v.x());
            vertex_data.push_back(v.y());
            vertex_data.push_back(v.z());
        }

        // Prepare face data for GPU (assuming triangular faces)
        std::vector<int> face_data;
        face_data.reserve(faces.size() * 3);
        for (const auto& face : faces) {
            if (face.size() >= 3) {
                face_data.push_back(face[0]);
                face_data.push_back(face[1]);
                face_data.push_back(face[2]);
            }
        }

        // Allocate GPU memory for mesh data
        double* d_vertices;
        int* d_faces;
        cudaMalloc(&d_vertices, vertex_data.size() * sizeof(double));
        cudaMalloc(&d_faces, face_data.size() * sizeof(int));

        // Copy data to GPU
        cudaMemcpy(d_vertices, vertex_data.data(), vertex_data.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_faces, face_data.data(), face_data.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Define block and grid dimensions
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);

        // Launch mesh voxelization kernel
        voxelizeMeshKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            d_vertices, vertices.size(), d_faces, face_data.size() / 3, cost_value);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }

        // Free GPU memory
        cudaFree(d_vertices);
        cudaFree(d_faces);
#endif
    }

    int GPUCudaVoxelization::voxelizeEllipsoid(const std::shared_ptr<EllipsoidEntity>& ellipsoid, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        double center_x = ellipsoid->getProperties()["center_x"];
        double center_y = ellipsoid->getProperties()["center_y"];
        double center_z = ellipsoid->getProperties()["center_z"];
        double radius_x = ellipsoid->getProperties()["radius_x"] + buffer_size;
        double radius_y = ellipsoid->getProperties()["radius_y"] + buffer_size;
        double radius_z = ellipsoid->getProperties()["radius_z"] + buffer_size;
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);
        voxelizeEllipsoidKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, radius_x, radius_y, radius_z, cost_value);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return -1;
        }
        cudaDeviceSynchronize();
        copyFromGPU();
        int marked_voxels = 0;
        for (unsigned char voxel : voxel_grid_) {
            if (voxel == cost_value) marked_voxels++;
        }
        return marked_voxels;
#else
        return 0;
#endif
    }

    void GPUCudaVoxelization::voxelizeEllipsoidGPUOnly(const std::shared_ptr<EllipsoidEntity>& ellipsoid, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        double center_x = ellipsoid->getProperties()["center_x"];
        double center_y = ellipsoid->getProperties()["center_y"];
        double center_z = ellipsoid->getProperties()["center_z"];
        double radius_x = ellipsoid->getProperties()["radius_x"] + buffer_size;
        double radius_y = ellipsoid->getProperties()["radius_y"] + buffer_size;
        double radius_z = ellipsoid->getProperties()["radius_z"] + buffer_size;
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);
        voxelizeEllipsoidKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, radius_x, radius_y, radius_z, cost_value);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }
#endif
    }

    int GPUCudaVoxelization::voxelizeCone(const std::shared_ptr<ConeEntity>& cone, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        double center_x = cone->getProperties()["center_x"];
        double center_y = cone->getProperties()["center_y"];
        double center_z = cone->getProperties()["center_z"];
        double radius = cone->getProperties()["radius"] + buffer_size;
        double height = cone->getProperties()["height"] + 2 * buffer_size;
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);
        voxelizeConeKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, radius, height, cost_value);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return -1;
        }
        cudaDeviceSynchronize();
        copyFromGPU();
        int marked_voxels = 0;
        for (unsigned char voxel : voxel_grid_) {
            if (voxel == cost_value) marked_voxels++;
        }
        return marked_voxels;
#else
        return 0;
#endif
    }

    void GPUCudaVoxelization::voxelizeConeGPUOnly(const std::shared_ptr<ConeEntity>& cone, double buffer_size, unsigned char cost_value) {
#ifdef ENABLE_CUDA
        double center_x = cone->getProperties()["center_x"];
        double center_y = cone->getProperties()["center_y"];
        double center_z = cone->getProperties()["center_z"];
        double radius = cone->getProperties()["radius"] + buffer_size;
        double height = cone->getProperties()["height"] + 2 * buffer_size;
        dim3 blockSize(16, 16, 4);
        dim3 gridSize((grid_x_ + blockSize.x - 1) / blockSize.x,
            (grid_y_ + blockSize.y - 1) / blockSize.y,
            (grid_z_ + blockSize.z - 1) / blockSize.z);
        voxelizeConeKernel << <gridSize, blockSize >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            resolution_xy_, resolution_z_, origin_x_, origin_y_, origin_z_,
            center_x, center_y, center_z, radius, height, cost_value);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }
#endif
    }

    bool GPUCudaVoxelization::worldToVoxel(double world_x, double world_y, double world_z,
        int& voxel_x, int& voxel_y, int& voxel_z) const {
        voxel_x = static_cast<int>((world_x - origin_x_) / resolution_xy_);
        voxel_y = static_cast<int>((world_y - origin_y_) / resolution_xy_);
        voxel_z = static_cast<int>((world_z - origin_z_) / resolution_z_);
        return isValidVoxel(voxel_x, voxel_y, voxel_z);
    }

    void GPUCudaVoxelization::voxelToWorld(int voxel_x, int voxel_y, int voxel_z,
        double& world_x, double& world_y, double& world_z) const {
        world_x = origin_x_ + (voxel_x + 0.5) * resolution_xy_;
        world_y = origin_y_ + (voxel_y + 0.5) * resolution_xy_;
        world_z = origin_z_ + (voxel_z + 0.5) * resolution_z_;
    }

    bool GPUCudaVoxelization::isValidVoxel(int voxel_x, int voxel_y, int voxel_z) const {
        return voxel_x >= 0 && voxel_x < grid_x_ &&
            voxel_y >= 0 && voxel_y < grid_y_ &&
            voxel_z >= 0 && voxel_z < grid_z_;
    }

    void GPUCudaVoxelization::updateVoxelCost(int voxel_x, int voxel_y, int voxel_z, unsigned char cost_value) {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            voxel_grid_[index] = cost_value;
        }
    }

    unsigned char GPUCudaVoxelization::getVoxelCost(int voxel_x, int voxel_y, int voxel_z) const {
        if (isValidVoxel(voxel_x, voxel_y, voxel_z)) {
            int index = voxel_z * grid_x_ * grid_y_ + voxel_y * grid_x_ + voxel_x;
            return voxel_grid_[index];
        }
        return 0;
    }

    bool GPUCudaVoxelization::saveToFile(const std::string& filename) const {
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

    bool GPUCudaVoxelization::loadFromFile(const std::string& filename) {
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

    bool GPUCudaVoxelization::saveVoxelGrid(const std::string& filename) const {
        return saveToFile(filename);
    }

    void GPUCudaVoxelization::clear() {
        std::fill(voxel_grid_.begin(), voxel_grid_.end(), 0);
        if (gpu_available_ && d_voxel_grid_ != nullptr) {
            cudaMemset(d_voxel_grid_, 0, grid_size_);
        }
    }

    void GPUCudaVoxelization::allocateGPUMemory() {
#ifdef ENABLE_CUDA
        if (d_voxel_grid_ != nullptr) {
            cudaFree(d_voxel_grid_);
        }
        cudaMalloc(&d_voxel_grid_, grid_size_);
        cudaMemset(d_voxel_grid_, 0, grid_size_);
#endif
    }

    void GPUCudaVoxelization::freeGPUMemory() {
#ifdef ENABLE_CUDA
        if (d_voxel_grid_ != nullptr) {
            cudaFree(d_voxel_grid_);
            d_voxel_grid_ = nullptr;
        }
#endif
    }

    void GPUCudaVoxelization::copyToGPU() {
#ifdef ENABLE_CUDA
        if (gpu_available_ && d_voxel_grid_ != nullptr) {
            cudaMemcpy(d_voxel_grid_, voxel_grid_.data(), grid_size_, cudaMemcpyHostToDevice);
        }
#endif
    }

    void GPUCudaVoxelization::copyFromGPU() {
#ifdef ENABLE_CUDA
        if (gpu_available_ && d_voxel_grid_ != nullptr) {
            cudaMemcpy(voxel_grid_.data(), d_voxel_grid_, grid_size_, cudaMemcpyDeviceToHost);
        }
#endif
    }

    void GPUCudaVoxelization::setResolution(int x, int y, int z) {
        grid_x_ = x;
        grid_y_ = y;
        grid_z_ = z;
        voxel_grid_.resize(grid_x_ * grid_y_ * grid_z_, 0);
        grid_size_ = voxel_grid_.size();
        if (gpu_available_) {
            allocateGPUMemory();
        }
    }

    void GPUCudaVoxelization::setBoundingBox(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z) {
        origin_x_ = min_x;
        origin_y_ = min_y;
        origin_z_ = min_z;
        resolution_xy_ = (max_x - min_x) / grid_x_;
        resolution_z_ = (max_z - min_z) / grid_z_;
    }

    int GPUCudaVoxelization::voxelizeBoxGPU(const std::shared_ptr<SpatialEntity>& entity, double buffer_size, unsigned char cost_value) {
        auto box = std::dynamic_pointer_cast<BoxEntity>(entity);
        if (!box) return 0;
        return voxelizeBox(box, buffer_size, cost_value);
    }

    int GPUCudaVoxelization::voxelizeSphereGPU(const std::shared_ptr<SpatialEntity>& entity, double buffer_size, unsigned char cost_value) {
        auto sphere = std::dynamic_pointer_cast<SphereEntity>(entity);
        if (!sphere) return 0;
        return voxelizeSphere(sphere, buffer_size, cost_value);
    }

    int GPUCudaVoxelization::voxelizeCylinderGPU(const std::shared_ptr<SpatialEntity>& entity, double buffer_size, unsigned char cost_value) {
        auto cylinder = std::dynamic_pointer_cast<CylinderEntity>(entity);
        if (!cylinder) return 0;
        return voxelizeCylinder(cylinder, buffer_size, cost_value);
    }

    bool GPUCudaVoxelization::voxelize(const std::vector<std::shared_ptr<SpatialEntity>>& entities) {
        clear();
        return voxelizeEntities(entities) > 0;
    }

    void GPUCudaVoxelization::getVoxelGrid(std::vector<unsigned char>& out) const {
        out = voxel_grid_;
    }
        } // namespace voxelization
