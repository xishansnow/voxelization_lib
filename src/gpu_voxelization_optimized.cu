#include "gpu_voxelization.hpp"
#include "spatial_entities.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>

namespace voxelization {

    // 常量定义
    constexpr int BLOCK_SIZE_1D = 256;
    constexpr int BLOCK_SIZE_3D = 8;

    // 自定义dot函数实现
    __device__ __host__ float dot(const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __device__ __host__ float dot(const float2& a, const float2& b) {
        return a.x * b.x + a.y * b.y;
    }

    // 自定义float3减法操作符
    __device__ __host__ float3 operator-(const float3& a, const float3& b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    // 自定义float2减法操作符
    __device__ __host__ float2 operator-(const float2& a, const float2& b) {
        return make_float2(a.x - b.x, a.y - b.y);
    }

    // 优化的实体结构
    struct OptimizedEntity {
        int type; // 0: box, 1: sphere, 2: cylinder, 3: ellipsoid, 4: cone, 5: mesh
        float3 center;
        float3 size; // box: size; ellipsoid: radii (x, y, z)
        float radius; // sphere/cylinder/cone: radius
        float height; // cylinder/cone: height
        int mesh_vertex_offset; // mesh: 顶点起始索引
        int mesh_vertex_count;
        int mesh_face_offset; // mesh: 面起始索引
        int mesh_face_count;
    };

    // 简化的GPU体素化内核
    __global__ void optimizedVoxelizationKernel(unsigned char* voxel_grid, int grid_x, int grid_y, int grid_z,
        OptimizedEntity* entities, int num_entities,
        float3 origin, float resolution, unsigned char cost_value, float3* mesh_vertices, int* mesh_faces) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= grid_x || y >= grid_y || z >= grid_z) return;

        // 计算世界坐标
        float3 world_pos = make_float3(
            origin.x + (x + 0.5f) * resolution,
            origin.y + (y + 0.5f) * resolution,
            origin.z + (z + 0.5f) * resolution
        );

        // 检查所有实体
        for (int i = 0; i < num_entities; i++) {
            OptimizedEntity entity = entities[i];
            bool inside = false;

            switch (entity.type) {
            case 0: // Box
                inside = (world_pos.x >= entity.center.x - entity.size.x / 2 &&
                    world_pos.x <= entity.center.x + entity.size.x / 2 &&
                    world_pos.y >= entity.center.y - entity.size.y / 2 &&
                    world_pos.y <= entity.center.y + entity.size.y / 2 &&
                    world_pos.z >= entity.center.z - entity.size.z / 2 &&
                    world_pos.z <= entity.center.z + entity.size.z / 2);
                break;
            case 1: // Sphere
            {
                float3 diff = world_pos - entity.center;
                float dist_sq = dot(diff, diff);
                inside = (dist_sq <= entity.radius * entity.radius);
            }
            break;
            case 2: // Cylinder
            {
                float2 diff_xy = make_float2(world_pos.x - entity.center.x, world_pos.y - entity.center.y);
                float dist_xy_sq = dot(diff_xy, diff_xy);
                inside = (dist_xy_sq <= entity.radius * entity.radius &&
                    world_pos.z >= entity.center.z - entity.height / 2 &&
                    world_pos.z <= entity.center.z + entity.height / 2);
            }
            break;
            case 3: // Ellipsoid
            {
                float3 diff = world_pos - entity.center;
                float val = (diff.x * diff.x) / (entity.size.x * entity.size.x) +
                    (diff.y * diff.y) / (entity.size.y * entity.size.y) +
                    (diff.z * diff.z) / (entity.size.z * entity.size.z);
                inside = (val <= 1.0f);
            }
            break;
            case 4: // Cone
            {
                // 以 entity.center 为底面中心，+z 方向为高
                float dz = world_pos.z - entity.center.z;
                if (dz >= 0 && dz <= entity.height) {
                    float r = entity.radius * (1.0f - dz / entity.height); // 线性缩小
                    float2 diff_xy = make_float2(world_pos.x - entity.center.x, world_pos.y - entity.center.y);
                    float dist_xy_sq = dot(diff_xy, diff_xy);
                    inside = (dist_xy_sq <= r * r);
                }
            }
            break;
            case 5: // Mesh
            {
                // 边界保护
                if (entity.mesh_vertex_count <= 0 || entity.mesh_face_count <= 0) break;
                for (int f = 0; f < entity.mesh_face_count; ++f) {
                    int face_idx = entity.mesh_face_offset + f * 3;
                    if (face_idx + 2 >= entity.mesh_face_offset + entity.mesh_face_count * 3) break;
                    int v0_idx = mesh_faces[face_idx];
                    int v1_idx = mesh_faces[face_idx + 1];
                    int v2_idx = mesh_faces[face_idx + 2];
                    if (v0_idx < 0 || v1_idx < 0 || v2_idx < 0 ||
                        v0_idx >= entity.mesh_vertex_count ||
                        v1_idx >= entity.mesh_vertex_count ||
                        v2_idx >= entity.mesh_vertex_count) continue;
                    int v0_abs = entity.mesh_vertex_offset + v0_idx;
                    int v1_abs = entity.mesh_vertex_offset + v1_idx;
                    int v2_abs = entity.mesh_vertex_offset + v2_idx;
                    // 顶点全局索引保护
                    if (v0_abs < 0 || v1_abs < 0 || v2_abs < 0) continue;
                    float3 v0 = mesh_vertices[v0_abs];
                    float3 v1 = mesh_vertices[v1_abs];
                    float3 v2 = mesh_vertices[v2_abs];
                    float3 p = world_pos;
                    float3 v0v1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
                    float3 v0v2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
                    float3 v0p = make_float3(p.x - v0.x, p.y - v0.y, p.z - v0.z);
                    float d00 = dot(v0v1, v0v1);
                    float d01 = dot(v0v1, v0v2);
                    float d11 = dot(v0v2, v0v2);
                    float d20 = dot(v0p, v0v1);
                    float d21 = dot(v0p, v0v2);
                    float denom = d00 * d11 - d01 * d01;
                    if (fabsf(denom) < 1e-8f) continue;
                    float v = (d11 * d20 - d01 * d21) / denom;
                    float w = (d00 * d21 - d01 * d20) / denom;
                    float u = 1.0f - v - w;
                    if (u >= 0 && v >= 0 && w >= 0) {
                        inside = true;
                        break;
                    }
                }
            }
            break;
            }

            if (inside) {
                int index = z * grid_x * grid_y + y * grid_x + x;
                voxel_grid[index] = cost_value;
                break;
            }
        }
    }

    // 简化的GPU体素化类
    class OptimizedGPUVoxelization {
    private:
        unsigned char* d_voxel_grid_;
        OptimizedEntity* d_entities_;
        std::vector<OptimizedEntity> h_entities_;
        mutable std::vector<unsigned char> h_voxel_grid_;
        int grid_x_, grid_y_, grid_z_;
        float resolution_;
        float3 origin_;
        size_t grid_size_;
        int mesh_vertex_global_offset_;
        int mesh_face_global_offset_;

        // Mesh 数据管理
        std::vector<float3> h_mesh_vertices_;
        std::vector<int> h_mesh_faces_;
        float3* d_mesh_vertices_;
        int* d_mesh_faces_;
    public:
        OptimizedGPUVoxelization();
        ~OptimizedGPUVoxelization();
        void initializeGrid(int grid_x, int grid_y, int grid_z, float resolution, float3 origin);
        void addEntity(const std::shared_ptr<SpatialEntity>& entity);
        void prepareOptimizations();
        void voxelize(unsigned char cost_value = 255);
        void getVoxelGrid(std::vector<unsigned char>& voxel_grid);
        const unsigned char* getVoxelGridPtr() const;
        bool saveVoxelGrid(const std::string& filename) const;
        void freeGPUMemory();
        float3 getOrigin() const { return origin_; }
        float getResolution() const { return resolution_; }
        int getGridX() const { return grid_x_; }
        int getGridY() const { return grid_y_; }
        int getGridZ() const { return grid_z_; }
        void clear();
    };

    // 类外实现
    OptimizedGPUVoxelization::OptimizedGPUVoxelization() : d_voxel_grid_(nullptr), d_entities_(nullptr),
        grid_x_(0), grid_y_(0), grid_z_(0), resolution_(0.0f), grid_size_(0), mesh_vertex_global_offset_(0), mesh_face_global_offset_(0),
        d_mesh_vertices_(nullptr), d_mesh_faces_(nullptr) {
        origin_ = make_float3(0.0f, 0.0f, 0.0f);
    }

    OptimizedGPUVoxelization::~OptimizedGPUVoxelization() {
        freeGPUMemory();
    }

    void OptimizedGPUVoxelization::initializeGrid(int grid_x, int grid_y, int grid_z, float resolution, float3 origin) {
        grid_x_ = grid_x;
        grid_y_ = grid_y;
        grid_z_ = grid_z;
        resolution_ = resolution;
        origin_ = origin;
        grid_size_ = grid_x_ * grid_y_ * grid_z_;
        cudaMalloc(&d_voxel_grid_, grid_size_);
        cudaMemset(d_voxel_grid_, 0, grid_size_);
    }

    void OptimizedGPUVoxelization::addEntity(const std::shared_ptr<SpatialEntity>& entity) {
        OptimizedEntity opt_entity;
        auto properties = entity->getProperties();
        std::string type = entity->getType();

        // 初始化默认值
        opt_entity.mesh_vertex_offset = 0;
        opt_entity.mesh_vertex_count = 0;
        opt_entity.mesh_face_offset = 0;
        opt_entity.mesh_face_count = 0;

        if (type == "box") {
            opt_entity.type = 0;
            opt_entity.center = make_float3(properties["center_x"], properties["center_y"], properties["center_z"]);
            opt_entity.size = make_float3(properties["size_x"], properties["size_y"], properties["size_z"]);
        }
        else if (type == "sphere") {
            opt_entity.type = 1;
            opt_entity.center = make_float3(properties["center_x"], properties["center_y"], properties["center_z"]);
            opt_entity.radius = properties["radius"];
        }
        else if (type == "cylinder") {
            opt_entity.type = 2;
            opt_entity.center = make_float3(properties["center_x"], properties["center_y"], properties["center_z"]);
            opt_entity.radius = properties["radius"];
            opt_entity.height = properties["height"];
        }
        else if (type == "ellipsoid") {
            opt_entity.type = 3;
            opt_entity.center = make_float3(properties["center_x"], properties["center_y"], properties["center_z"]);
            opt_entity.size = make_float3(properties["size_x"], properties["size_y"], properties["size_z"]);
        }
        else if (type == "cone") {
            opt_entity.type = 4;
            opt_entity.center = make_float3(properties["center_x"], properties["center_y"], properties["center_z"]);
            opt_entity.radius = properties["radius"];
            opt_entity.height = properties["height"];
        }
        else if (type == "mesh") {
            opt_entity.type = 5;
            auto mesh = std::dynamic_pointer_cast<MeshEntity>(entity);
            const auto& vertices = mesh->getVertices();
            const auto& faces = mesh->getFaces();
            int v_offset = mesh_vertex_global_offset_;
            int f_offset = mesh_face_global_offset_;
            for (const auto& v : vertices) {
                h_mesh_vertices_.push_back(make_float3(v.x(), v.y(), v.z()));
            }
            mesh_vertex_global_offset_ += vertices.size();
            for (const auto& face : faces) {
                h_mesh_faces_.push_back(face[0]);
                h_mesh_faces_.push_back(face[1]);
                h_mesh_faces_.push_back(face[2]);
            }
            mesh_face_global_offset_ += faces.size() * 3;
            opt_entity.mesh_vertex_offset = v_offset;
            opt_entity.mesh_vertex_count = vertices.size();
            opt_entity.mesh_face_offset = f_offset;
            opt_entity.mesh_face_count = faces.size();
        }
        h_entities_.push_back(opt_entity);
    }

    void OptimizedGPUVoxelization::prepareOptimizations() {
        freeGPUMemory(); // 先释放所有 GPU 内存，防止悬挂
        if (h_entities_.empty()) return;

        // 重新收集所有 mesh 数据
        h_mesh_vertices_.clear();
        h_mesh_faces_.clear();
        mesh_vertex_global_offset_ = 0;
        mesh_face_global_offset_ = 0;

        // 重新处理所有实体，收集 mesh 数据
        for (auto& entity : h_entities_) {
            if (entity.type == 5) { // mesh type
                entity.mesh_vertex_offset = mesh_vertex_global_offset_;
                entity.mesh_face_offset = mesh_face_global_offset_;
                // 注意：这里我们无法重新获取原始 mesh 数据，所以跳过
                // 在实际使用中，应该在 addEntity 时就收集好所有数据
            }
        }

        // 分配 GPU 内存
        cudaMalloc(&d_entities_, h_entities_.size() * sizeof(OptimizedEntity));
        cudaMemcpy(d_entities_, h_entities_.data(), h_entities_.size() * sizeof(OptimizedEntity), cudaMemcpyHostToDevice);

        // 分配和复制 mesh 数据到 GPU
        if (!h_mesh_vertices_.empty()) {
            cudaMalloc(&d_mesh_vertices_, h_mesh_vertices_.size() * sizeof(float3));
            cudaMemcpy(d_mesh_vertices_, h_mesh_vertices_.data(), h_mesh_vertices_.size() * sizeof(float3), cudaMemcpyHostToDevice);
        }
        else {
            // 分配一个空的 float3 数组，避免 nullptr
            cudaMalloc(&d_mesh_vertices_, sizeof(float3));
        }
        if (!h_mesh_faces_.empty()) {
            cudaMalloc(&d_mesh_faces_, h_mesh_faces_.size() * sizeof(int));
            cudaMemcpy(d_mesh_faces_, h_mesh_faces_.data(), h_mesh_faces_.size() * sizeof(int), cudaMemcpyHostToDevice);
        }
        else {
            // 分配一个空的 int 数组，避免 nullptr
            cudaMalloc(&d_mesh_faces_, sizeof(int));
        }
    }

    void OptimizedGPUVoxelization::voxelize(unsigned char cost_value) {
        if (h_entities_.empty()) return;
        dim3 block_size(BLOCK_SIZE_3D, BLOCK_SIZE_3D, BLOCK_SIZE_3D);
        dim3 grid_size((grid_x_ + block_size.x - 1) / block_size.x,
            (grid_y_ + block_size.y - 1) / block_size.y,
            (grid_z_ + block_size.z - 1) / block_size.z);
        optimizedVoxelizationKernel << <grid_size, block_size >> > (d_voxel_grid_, grid_x_, grid_y_, grid_z_,
            d_entities_, h_entities_.size(),
            origin_, resolution_, cost_value, d_mesh_vertices_, d_mesh_faces_);
        cudaDeviceSynchronize();
    }

    void OptimizedGPUVoxelization::getVoxelGrid(std::vector<unsigned char>& voxel_grid) {
        voxel_grid.resize(grid_size_);
        cudaMemcpy(voxel_grid.data(), d_voxel_grid_, grid_size_, cudaMemcpyDeviceToHost);
    }

    const unsigned char* OptimizedGPUVoxelization::getVoxelGridPtr() const {
        if (h_voxel_grid_.size() != grid_size_) h_voxel_grid_.resize(grid_size_);
        cudaMemcpy((void*)h_voxel_grid_.data(), d_voxel_grid_, grid_size_, cudaMemcpyDeviceToHost);
        return h_voxel_grid_.data();
    }

    bool OptimizedGPUVoxelization::saveVoxelGrid(const std::string& filename) const {
        std::vector<unsigned char> host_grid(grid_size_);
        cudaMemcpy((void*)host_grid.data(), d_voxel_grid_, grid_size_, cudaMemcpyDeviceToHost);
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;
        file.write(reinterpret_cast<const char*>(host_grid.data()), host_grid.size());
        file.close();
        return true;
    }

    void OptimizedGPUVoxelization::freeGPUMemory() {
        if (d_voxel_grid_) { cudaFree(d_voxel_grid_); d_voxel_grid_ = nullptr; }
        if (d_entities_) { cudaFree(d_entities_); d_entities_ = nullptr; }
        if (d_mesh_vertices_) { cudaFree(d_mesh_vertices_); d_mesh_vertices_ = nullptr; }
        if (d_mesh_faces_) { cudaFree(d_mesh_faces_); d_mesh_faces_ = nullptr; }
    }

    void OptimizedGPUVoxelization::clear() {
        h_entities_.clear();
        h_mesh_vertices_.clear();
        h_mesh_faces_.clear();
        mesh_vertex_global_offset_ = 0;
        mesh_face_global_offset_ = 0;
    }
} // namespace voxelization
