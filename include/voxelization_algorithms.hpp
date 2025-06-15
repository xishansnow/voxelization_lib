#ifndef VOXELIZATION_ALGORITHMS_HPP
#define VOXELIZATION_ALGORITHMS_HPP

#include "voxelization_base.hpp"
#include "cpu_voxelization.hpp"
#include "gpu_voxelization.hpp"
#include "hybrid_voxelization.hpp"
#include "mesh_loader.hpp"
#include <vector>
#include <memory>
#include <fstream>

namespace voxelization {

    // Forward declarations for backward compatibility
    class CPUSequentialVoxelization;
    class CPUParallelVoxelization;
    class GPUCudaVoxelization;
    class HybridVoxelization;

} // namespace voxelization

#endif // VOXELIZATION_ALGORITHMS_HPP
