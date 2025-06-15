# Voxelization Library

一个独立的C++体素化库，专门用于实现各种体素化算法，支持多种空间实体类型和多种体素化算法。

## 特性

- **多种空间实体支持**：盒子、圆柱体、球体、椭球体、圆锥体、网格、复合实体
- **多种体素化算法**：CPU顺序、CPU并行（OpenMP）、GPU（CUDA）、混合算法
- **高性能**：支持OpenMP并行化和CUDA GPU加速
- **易用性**：工厂模式创建算法和实体，简洁的API
- **可扩展性**：模块化设计，易于添加新的实体类型和算法
- **文件I/O**：支持体素网格的保存和加载

## 依赖

- C++17 或更高版本
- CMake 3.8 或更高版本
- Eigen3
- OpenMP（可选，用于并行算法）
- CUDA（可选，用于GPU算法）

## 安装

### 从源码编译

```bash
# 克隆仓库
git clone <repository-url>
cd voxelization_lib

# 创建构建目录
mkdir build && cd build

# 配置
cmake ..

# 编译
make -j$(nproc)

# 安装（可选）
sudo make install
```

### 启用CUDA支持

```bash
cmake -DENABLE_CUDA=ON ..
```

## 使用方法

### 基本用法

```cpp
#include "voxelization_factory.hpp"
#include "spatial_entities.hpp"

using namespace voxelization;

int main() {
    // 1. 创建空间实体
    auto box = std::make_shared<BoxEntity>(10.0, 10.0, 2.0, 4.0, 4.0, 4.0);
    auto sphere = std::make_shared<SphereEntity>(15.0, 5.0, 1.0, 1.5);
    
    // 2. 创建体素化算法
    auto voxelizer = VoxelizationFactory::createAlgorithm(
        VoxelizationFactory::AlgorithmType::CPU_PARALLEL);
    
    // 3. 初始化体素网格
    voxelizer->initialize(200, 200, 100, 0.1, 0.1, 0.0, 0.0, 0.0);
    
    // 4. 体素化实体
    int voxels = voxelizer->voxelizeEntity(box, 0.2, 255);
    
    // 5. 保存结果
    voxelizer->saveToFile("output.bin");
    
    return 0;
}
```

### 高级用法

```cpp
// 创建复合实体
std::vector<std::shared_ptr<SpatialEntity>> entities = {box, sphere, cylinder};
auto composite = std::make_shared<CompositeEntity>(entities);

// 体素化多个实体
std::vector<std::shared_ptr<SpatialEntity>> all_entities = {box, sphere, cylinder, composite};
int total_voxels = voxelizer->voxelizeEntities(all_entities, 0.1, 255);

// 坐标转换
int voxel_x, voxel_y, voxel_z;
if (voxelizer->worldToVoxel(10.0, 10.0, 2.0, voxel_x, voxel_y, voxel_z)) {
    unsigned char cost = voxelizer->getVoxelCost(voxel_x, voxel_y, voxel_z);
    std::cout << "Voxel cost: " << static_cast<int>(cost) << std::endl;
}
```

## API 参考

### 空间实体

#### BoxEntity
```cpp
BoxEntity(double center_x, double center_y, double center_z,
          double size_x, double size_y, double size_z);
```

#### CylinderEntity
```cpp
CylinderEntity(double center_x, double center_y, double center_z,
               double radius, double height);
```

#### SphereEntity
```cpp
SphereEntity(double center_x, double center_y, double center_z, double radius);
```

#### EllipsoidEntity
```cpp
EllipsoidEntity(double center_x, double center_y, double center_z,
                double radius_x, double radius_y, double radius_z);
```

#### ConeEntity
```cpp
ConeEntity(double center_x, double center_y, double center_z,
           double radius, double height);
```

#### MeshEntity
```cpp
MeshEntity(const std::vector<Eigen::Vector3d>& vertices,
           const std::vector<std::vector<int>>& faces);
```

#### CompositeEntity
```cpp
CompositeEntity(const std::vector<std::shared_ptr<SpatialEntity>>& entities);
```

### 体素化算法

#### 算法类型
- `CPU_SEQUENTIAL`: CPU顺序算法
- `CPU_PARALLEL`: CPU并行算法（OpenMP）
- `GPU_CUDA`: GPU算法（CUDA）
- `HYBRID`: 混合算法

#### 主要方法
```cpp
// 初始化
void initialize(int grid_x, int grid_y, int grid_z,
               double resolution_xy, double resolution_z,
               double origin_x = 0.0, double origin_y = 0.0, double origin_z = 0.0);

// 体素化单个实体
int voxelizeEntity(const std::shared_ptr<SpatialEntity>& entity,
                  double buffer_size = 0.0,
                  unsigned char cost_value = 255);

// 体素化多个实体
int voxelizeEntities(const std::vector<std::shared_ptr<SpatialEntity>>& entities,
                    double buffer_size = 0.0,
                    unsigned char cost_value = 255);

// 坐标转换
bool worldToVoxel(double world_x, double world_y, double world_z,
                 int& voxel_x, int& voxel_y, int& voxel_z) const;

void voxelToWorld(int voxel_x, int voxel_y, int voxel_z,
                 double& world_x, double& world_y, double& world_z) const;

// 体素访问
unsigned char getVoxelCost(int voxel_x, int voxel_y, int voxel_z) const;
void updateVoxelCost(int voxel_x, int voxel_y, int voxel_z, unsigned char cost_value);

// 文件I/O
bool saveToFile(const std::string& filename) const;
bool loadFromFile(const std::string& filename);
```

## 性能优化

### CPU并行化
库自动使用OpenMP进行并行化，可以通过设置环境变量控制线程数：
```bash
export OMP_NUM_THREADS=8
```

### GPU加速
启用CUDA支持后，大型体素网格会自动使用GPU加速：
```cpp
auto gpu_voxelizer = VoxelizationFactory::createAlgorithm(
    VoxelizationFactory::AlgorithmType::GPU_CUDA);
```

### 混合算法
混合算法会根据实体大小和数量自动选择最优策略：
```cpp
auto hybrid_voxelizer = VoxelizationFactory::createAlgorithm(
    VoxelizationFactory::AlgorithmType::HYBRID);
```

## 测试

运行测试程序：
```bash
cd build
./test_voxelization
```

运行示例程序：
```bash
cd build
./example_usage
```

## 集成到ROS项目

### 方法1：作为外部依赖

在您的ROS包的`CMakeLists.txt`中添加：
```cmake
find_package(voxelization_lib REQUIRED)
target_link_libraries(your_node voxelization_lib::voxelization_lib)
```

### 方法2：直接包含

将库文件复制到您的项目中，并在`CMakeLists.txt`中添加：
```cmake
add_subdirectory(voxelization_lib)
target_link_libraries(your_node voxelization_lib)
```

## 文件格式

### 体素网格文件格式
体素网格以二进制格式保存，包含：
- 网格尺寸（3个int）
- 分辨率（2个double）
- 原点坐标（3个double）
- 体素数据（grid_x * grid_y * grid_z个unsigned char）

## 贡献

欢迎提交问题和拉取请求！

## 许可证

MIT License

## 更新日志

### v1.0.0
- 初始版本
- 支持基本空间实体类型
- 实现CPU顺序和并行算法
- 添加文件I/O功能
- 提供完整的测试和示例 