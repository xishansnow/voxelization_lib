# 性能基准测试程序使用说明

本目录包含了用于比较CPU串行、CPU并行和GPU方法效率的测试程序。

## 程序说明

### 1. 基础性能测试程序 (`performance_benchmark.cpp`)

这是一个基础的性能测试程序，用于快速比较不同算法的性能。

**功能特点：**
- 生成随机空间实体进行测试
- 支持命令行参数配置
- 输出详细的性能指标
- 生成CSV格式的结果文件

**使用方法：**
```bash
# 编译
cd build
make performance_benchmark

# 运行（使用默认参数）
./performance_benchmark

# 运行（自定义参数）
./performance_benchmark [实体数量] [网格X] [网格Y] [网格Z] [分辨率XY] [分辨率Z]

# 示例：测试500个实体，网格大小300x300x150，分辨率0.05
./performance_benchmark 500 300 300 150 0.05 0.05
```

**输出结果：**
- 控制台显示详细的性能对比表格
- 生成 `benchmark_results.csv` 文件

### 2. 高级性能测试程序 (`advanced_benchmark.cpp`)

这是一个更全面的性能测试程序，包含多种测试场景和详细分析。

**功能特点：**
- 5种不同的测试场景（小规模、中规模、大规模、高密度、聚类）
- 支持聚类实体生成
- 详细的性能分析和速度比计算
- 生成综合报告

**使用方法：**
```bash
# 编译
cd build
make advanced_benchmark

# 运行
./advanced_benchmark
```

**输出结果：**
- 控制台显示每个场景的详细结果
- 生成 `advanced_benchmark_results.csv` 文件
- 生成 `benchmark_summary_report.txt` 报告文件

## 测试场景说明

### 基础测试程序场景
- 随机生成指定数量的空间实体
- 实体类型：盒子、圆柱体、球体、椭球体、圆锥体
- 位置和大小随机分布

### 高级测试程序场景

1. **Small Scale（小规模）**
   - 50个实体，100x100x50网格，分辨率0.2
   - 适用于快速验证和调试

2. **Medium Scale（中规模）**
   - 200个实体，200x200x100网格，分辨率0.1
   - 平衡性能和测试时间

3. **Large Scale（大规模）**
   - 500个实体，400x400x200网格，分辨率0.05
   - 测试算法在大规模数据下的性能

4. **High Density（高密度）**
   - 300个实体，300x300x150网格，分辨率0.08
   - 实体在小空间内密集分布

5. **Clustered（聚类）**
   - 400个实体，350x350x175网格，分辨率0.1
   - 实体按聚类分布，模拟真实场景

## 性能指标说明

### 时间指标
- **Total Time (ms)**: 总执行时间（毫秒）
- **Avg/Entity (ms)**: 每个实体的平均处理时间（毫秒）

### 吞吐量指标
- **Total Voxels**: 生成的体素总数
- **Voxels/sec**: 每秒处理的体素数量

### 资源指标
- **Memory (MB)**: 内存使用量（兆字节）
- **CPU Utilization**: CPU利用率估计

### 速度比分析
- 显示各算法相对于基准算法的性能提升倍数

## 结果文件格式

### CSV文件格式
```csv
Algorithm,Total_Time_ms,Avg_Time_per_Entity_ms,Total_Voxels,Voxels_per_Second,Memory_MB
CPU_SEQUENTIAL,1234.56,12.3456,1000000,810.23,4.00
CPU_PARALLEL,456.78,4.5678,1000000,2190.45,4.00
GPU_CUDA,234.56,2.3456,1000000,4265.32,4.00
```

### 高级测试CSV文件格式
```csv
Scenario,Algorithm,Total_Time_ms,Avg_Time_per_Entity_ms,Total_Voxels,Voxels_per_Second,Memory_MB,CPU_Utilization,Grid_Size_X,Grid_Size_Y,Grid_Size_Z,Resolution_XY,Resolution_Z
Medium Scale,CPU_SEQUENTIAL,1234.56,6.1728,1000000,810.23,4.00,100.0,200,200,100,0.1,0.1
```

## 环境要求

### 编译要求
- C++14 或更高版本
- CMake 3.5 或更高版本
- Eigen3
- OpenMP（可选，用于并行算法）
- CUDA（可选，用于GPU算法）

### 运行要求
- 支持OpenMP的系统（用于CPU并行算法）
- CUDA兼容的GPU和驱动程序（用于GPU算法）

## 故障排除

### 常见问题

1. **OpenMP不可用**
   - 错误信息：`OpenMP not found - parallel algorithms may not be available`
   - 解决方案：安装OpenMP开发包
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libomp-dev

   # CentOS/RHEL
   sudo yum install libomp-devel
   ```

2. **CUDA不可用**
   - 错误信息：`Error running benchmark for GPU_CUDA`
   - 解决方案：确保CUDA工具包正确安装并配置

3. **内存不足**
   - 对于大规模测试，可能需要增加系统内存或减少网格大小

### 性能优化建议

1. **CPU并行优化**
   - 设置合适的OpenMP线程数：
   ```bash
   export OMP_NUM_THREADS=8
   ```

2. **GPU优化**
   - 确保GPU驱动程序是最新版本
   - 监控GPU温度和功耗

3. **测试参数调整**
   - 根据系统性能调整实体数量和网格大小
   - 平衡测试时间和结果准确性

## 扩展和自定义

### 添加新的测试场景
在 `advanced_benchmark.cpp` 中的 `initializeTestScenarios()` 方法中添加新的测试场景配置。

### 添加新的实体类型
修改 `RandomEntityGenerator` 类以支持新的空间实体类型。

### 自定义性能指标
扩展 `BenchmarkResult` 结构体以包含额外的性能指标
