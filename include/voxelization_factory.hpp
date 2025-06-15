#ifndef VOXELIZATION_FACTORY_HPP
#define VOXELIZATION_FACTORY_HPP

#include "voxelization_base.hpp"
#include <memory>
#include <vector>
#include <string>

namespace voxelization {

/**
 * @brief Factory class for creating voxelization algorithms
 */
class VoxelizationFactory {
public:
    enum class AlgorithmType {
        CPU_SEQUENTIAL,
        CPU_PARALLEL,
        GPU_CUDA,
        HYBRID
    };
    
    /**
     * @brief Create a voxelization algorithm
     * @param type Algorithm type
     * @return Shared pointer to voxelization algorithm
     */
    static std::shared_ptr<VoxelizationBase> createAlgorithm(AlgorithmType type);
    
    /**
     * @brief Get available algorithm types
     * @return Vector of available algorithm types
     */
    static std::vector<std::string> getAvailableAlgorithms();
    
    /**
     * @brief Get algorithm type from string
     * @param name Algorithm name
     * @return Algorithm type
     */
    static AlgorithmType getAlgorithmType(const std::string& name);
    
    /**
     * @brief Get algorithm name from type
     * @param type Algorithm type
     * @return Algorithm name
     */
    static std::string getAlgorithmName(AlgorithmType type);
};

} // namespace voxelization

#endif // VOXELIZATION_FACTORY_HPP 