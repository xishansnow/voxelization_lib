#ifndef MESH_LOADER_HPP
#define MESH_LOADER_HPP

#include "spatial_entities.hpp"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>

namespace voxelization {

    /**
     * @brief Structure to hold mesh data
     */
    struct MeshData {
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3f> normals;
        std::vector<unsigned int> indices;
        std::vector<Eigen::Vector2f> tex_coords;
        Eigen::Vector3f bounding_box_min;
        Eigen::Vector3f bounding_box_max;
        std::string name;
    };

    /**
     * @brief Utility class for loading and processing 3D mesh files
     */
    class MeshLoader {
    public:
        MeshLoader();
        ~MeshLoader() = default;

        /**
         * @brief Load a 3D model file
         * @param filename Path to the 3D model file
         * @return Shared pointer to MeshEntity, or nullptr if loading failed
         */
        std::shared_ptr<MeshEntity> loadMesh(const std::string& filename);

        /**
         * @brief Load mesh data without creating an entity
         * @param filename Path to the 3D model file
         * @return MeshData structure, or empty if loading failed
         */
        MeshData loadMeshData(const std::string& filename);

        /**
         * @brief Get supported file formats
         * @return Vector of supported file extensions
         */
        std::vector<std::string> getSupportedFormats() const;

        /**
         * @brief Check if a file format is supported
         * @param extension File extension (e.g., ".obj", ".fbx")
         * @return True if format is supported
         */
        bool isFormatSupported(const std::string& extension) const;

        /**
         * @brief Get last error message
         * @return Error message string
         */
        std::string getLastError() const { return last_error_; }

        /**
         * @brief Set import flags for Assimp
         * @param flags Assimp post-processing flags
         */
        void setImportFlags(unsigned int flags) { import_flags_ = flags; }

        /**
         * @brief Get current import flags
         * @return Current import flags
         */
        unsigned int getImportFlags() const { return import_flags_; }

    private:
        Assimp::Importer importer_;
        std::string last_error_;
        unsigned int import_flags_;

        void setDefaultImportFlags();
        bool processScene(const aiScene* scene, MeshData& mesh_data);
        bool processMesh(const aiMesh* mesh, MeshData& mesh_data);
        void processNode(const aiNode* node, const aiScene* scene, MeshData& mesh_data);
        Eigen::Vector3f convertVector(const aiVector3D& vec) const;
        Eigen::Vector2f convertVector2D(const aiVector3D& vec) const;
    };

} // namespace voxelization

#endif // MESH_LOADER_HPP
