#include "voxelization_factory.hpp"
#include "voxelization_algorithms.hpp"
#include "cpu_voxelization.hpp"
#include "gpu_voxelization.hpp"
#include "gpu_voxelization_optimized.hpp"
#include "forceflow_voxelizer.hpp"
#include "hybrid_voxelization.hpp"
#include <memory>
#include <iostream>

namespace voxelization {

    std::shared_ptr<VoxelizationBase> VoxelizationFactory::createAlgorithm(AlgorithmType type) {
        switch (type) {
        case AlgorithmType::CPU_SEQUENTIAL:
            return std::make_shared<CPUSequentialVoxelization>();

        case AlgorithmType::CPU_PARALLEL:
            return std::make_shared<CPUParallelVoxelization>();

        case AlgorithmType::GPU_CUDA:
            return std::make_shared<GPUCudaVoxelization>();

        case AlgorithmType::GPU_OPTIMIZED:
            return std::make_shared<OptimizedGPUCudaVoxelization>();

        case AlgorithmType::FORCEFLOW:
            return std::make_shared<ForceFlowVoxelizer>();

        case AlgorithmType::HYBRID:
            return std::make_shared<HybridVoxelization>();

        default:
            std::cerr << "[VoxelizationFactory] Unknown algorithm type, using CPU_SEQUENTIAL as fallback." << std::endl;
            return std::make_shared<CPUSequentialVoxelization>();
        }
    }

    std::vector<std::string> VoxelizationFactory::getAvailableAlgorithms() {
        return {
            "CPU_SEQUENTIAL",
            "CPU_PARALLEL",
            "GPU_CUDA",
            "GPU_OPTIMIZED",
            "FORCEFLOW",
            "HYBRID"
        };
    }

    VoxelizationFactory::AlgorithmType VoxelizationFactory::getAlgorithmType(const std::string& name) {
        if (name == "CPU_SEQUENTIAL") return AlgorithmType::CPU_SEQUENTIAL;
        if (name == "CPU_PARALLEL") return AlgorithmType::CPU_PARALLEL;
        if (name == "GPU_CUDA") return AlgorithmType::GPU_CUDA;
        if (name == "GPU_OPTIMIZED") return AlgorithmType::GPU_OPTIMIZED;
        if (name == "FORCEFLOW") return AlgorithmType::FORCEFLOW;
        if (name == "HYBRID") return AlgorithmType::HYBRID;
        return AlgorithmType::CPU_SEQUENTIAL; // Default
    }

    std::string VoxelizationFactory::getAlgorithmName(AlgorithmType type) {
        switch (type) {
        case AlgorithmType::CPU_SEQUENTIAL: return "CPU_SEQUENTIAL";
        case AlgorithmType::CPU_PARALLEL: return "CPU_PARALLEL";
        case AlgorithmType::GPU_CUDA: return "GPU_CUDA";
        case AlgorithmType::GPU_OPTIMIZED: return "GPU_OPTIMIZED";
        case AlgorithmType::FORCEFLOW: return "FORCEFLOW";
        case AlgorithmType::HYBRID: return "HYBRID";
        default: return "UNKNOWN";
        }
    }

} // namespace voxelization
