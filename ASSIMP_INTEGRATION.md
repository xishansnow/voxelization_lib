# Assimp 集成 - 3D 文件体素化

本项目已集成 Assimp 开源库，支持加载各种 3D 模型文件格式并进行体素化处理。

## 支持的文件格式

Assimp 支持多种 3D 文件格式，包括但不限于：

- **OBJ** (.obj) - Wavefront Object
- **FBX** (.fbx) - Autodesk FBX
- **3DS** (.3ds) - 3D Studio Max
- **DAE** (.dae) - Collada
- **PLY** (.ply) - Stanford Polygon Library
- **STL** (.stl) - Stereolithography
- **BLEND** (.blend) - Blender
- **MAX** (.max) - 3D Studio Max
- **MA** (.ma) - Maya ASCII
- **MB** (.mb) - Maya Binary
- **LWS** (.lws) - LightWave Scene
- **LWO** (.lwo) - LightWave Object
- **MD2** (.md2) - Quake 2 Model
- **MD3** (.md3) - Quake 3 Model
- **MD5** (.md5) - Doom 3 Model
- **SMD** (.smd) - Valve Source Model
- **B3D** (.b3d) - Blitz3D Model
- **IRR** (.irr) - Irrlicht Scene
- **IRRMESH** (.irrmesh) - Irrlicht Mesh
- **TER** (.ter) - Terragen Terrain
- **Q3O** (.q3o) - Quake 3 BSP
- **Q3S** (.q3s) - Quake 3 BSP
- **AC** (.ac) - AC3D
- **MS3D** (.ms3d) - Milkshape 3D
- **COB** (.cob) - TrueSpace
- **SCN** (.scn) - TrueSpace
- **X** (.x) - DirectX
- **ASE** (.ase) - 3D Studio Max ASCII
- **GLTF** (.gltf) - GL Transmission Format
- **GLB** (.glb) - GL Transmission Format Binary

## 安装依赖

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install libassimp-dev
```

### CentOS/RHEL/Fedora
```bash
sudo yum install assimp-devel
# 或
sudo dnf install assimp-devel
```

### macOS
```bash
brew install assimp
```

### Windows
使用 vcpkg 或手动编译 Assimp：
```bash
vcpkg install assimp
```

## 使用方法

### 1. 基本用法

```cpp
#include "voxelization_algorithms.hpp"
#include "mesh_loader.hpp"

// 创建网格加载器
voxelization::MeshLoader loader;

// 加载 3D 模型文件
auto mesh_entity = loader.loadMesh("model.obj");
if (!mesh_entity) {
    std::cerr << "Failed to load mesh: " << loader.getLastError() << std::endl;
    return;
}

// 创建体素化算法
auto voxelizer = voxelization::VoxelizationFactory::createAlgorithm(
    voxelization::VoxelizationType::CPU_PARALLEL
);

// 设置参数
voxelizer->setResolution(64, 64, 64);
auto bbox = mesh_entity->getBoundingBox();
voxelizer->setBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);

// 执行体素化
std::vector<std::shared_ptr<voxelization::SpatialEntity>> entities;
entities.push_back(mesh_entity);
bool success = voxelizer->voxelize(entities);
```

### 2. 命令行示例

编译后可以使用提供的示例程序：

```bash
# 编译
./build.sh

# 运行网格体素化示例
./mesh_voxelization_example model.obj 64
```

### 3. 检查支持的文件格式

```cpp
voxelization::MeshLoader loader;
auto formats = loader.getSupportedFormats();
for (const auto& fmt : formats) {
    std::cout << "Supported: " << fmt << std::endl;
}

// 检查特定格式
if (loader.isFormatSupported(".obj")) {
    std::cout << "OBJ format is supported" << std::endl;
}
```

## 功能特性

### 1. 网格数据提取

- **顶点数据** - 3D 坐标
- **法向量** - 表面法向量
- **纹理坐标** - UV 映射
- **面索引** - 三角形面数据
- **包围盒** - 自动计算

### 2. 点内测试

使用射线投射算法进行精确的点内测试：

```cpp
bool inside = mesh_entity->isPointInside(x, y, z);
```

### 3. 网格属性

获取网格的详细属性：

```cpp
auto props = mesh_entity->getProperties();
std::cout << "Vertices: " << props["num_vertices"] << std::endl;
std::cout << "Triangles: " << props["num_triangles"] << std::endl;
std::cout << "Volume: " << props["volume"] << std::endl;
```

### 4. 导入选项

可以自定义 Assimp 的导入选项：

```cpp
voxelization::MeshLoader loader;
loader.setImportFlags(
    aiProcess_Triangulate |
    aiProcess_GenNormals |
    aiProcess_CalcTangentSpace |
    aiProcess_JoinIdenticalVertices |
    aiProcess_RemoveRedundantMaterials |
    aiProcess_FixInfacingNormals |
    aiProcess_ImproveCacheLocality |
    aiProcess_OptimizeMeshes |
    aiProcess_OptimizeGraph
);
```

## 性能优化

### 1. 内存优化

- 自动合并重复顶点
- 优化网格结构
- 移除冗余材质

### 2. 计算优化

- 并行处理大型网格
- 缓存友好的数据布局
- 优化的射线投射算法

### 3. 精度控制

- 可配置的浮点精度
- 容错处理
- 边界情况处理

## 错误处理

```cpp
voxelization::MeshLoader loader;
auto mesh = loader.loadMesh("invalid_file.obj");
if (!mesh) {
    std::cerr << "Error: " << loader.getLastError() << std::endl;
    return;
}
```

常见错误：
- 文件不存在
- 不支持的文件格式
- 损坏的文件
- 内存不足
- 无效的网格数据

## 示例输出

运行 `mesh_voxelization_example` 的典型输出：

```
=== 3D Mesh Voxelization Example ===
Loading file: model.obj
Voxel resolution: 64x64x64

Mesh Information:
  Vertices: 1234
  Triangles: 2468
  Normals: 1234
  Texture coordinates: 1234
  Bounding box: (-1.0, -1.0, -1.0) to (1.0, 1.0, 1.0)
  Volume: 8.0

Creating voxelization algorithm...
Performing voxelization...
Voxelization completed in 45 ms

Voxelization Results:
  Total voxels: 262144
  Occupied voxels: 15678
  Occupancy rate: 5.98%

Voxel grid saved to: model_voxels.raw

Testing point-in-mesh functionality...
  Center point (0.0, 0.0, 0.0) is inside the mesh
  Corner point (-2.0, -2.0, -2.0) is outside the mesh

=== Voxelization Complete ===
```

## 注意事项

1. **内存使用** - 大型模型可能需要大量内存
2. **精度** - 浮点精度可能影响复杂几何体的处理
3. **性能** - 复杂网格的体素化可能需要较长时间
4. **兼容性** - 某些文件格式可能有兼容性问题

## 故障排除

### 编译错误
- 确保已安装 Assimp 开发库
- 检查 CMake 配置
- 验证链接器设置

### 运行时错误
- 检查文件路径和权限
- 验证文件格式支持
- 查看错误消息

### 性能问题
- 降低体素分辨率
- 使用更简单的网格
- 启用优化选项
