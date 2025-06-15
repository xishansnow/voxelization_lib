#include "mesh_loader.hpp"
#include <assimp/importerdesc.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <assimp/version.h>

namespace voxelization {

    // MeshEntity implementation
    MeshEntity::MeshEntity(const std::string& filename) {
        loadFromFile(filename);
    }

    MeshEntity::MeshEntity(const MeshData& mesh_data) : mesh_data_(mesh_data) {
        computeBoundingBox();
    }

    std::vector<double> MeshEntity::getBoundingBox() const {
        return {
            mesh_data_.bounding_box_min.x(),
            mesh_data_.bounding_box_min.y(),
            mesh_data_.bounding_box_min.z(),
            mesh_data_.bounding_box_max.x(),
            mesh_data_.bounding_box_max.y(),
            mesh_data_.bounding_box_max.z()
        };
    }

    bool MeshEntity::isPointInside(double x, double y, double z) const {
        if (!isValid()) return false;

        // First check if point is inside bounding box
        if (x < mesh_data_.bounding_box_min.x() || x > mesh_data_.bounding_box_max.x() ||
            y < mesh_data_.bounding_box_min.y() || y > mesh_data_.bounding_box_max.y() ||
            z < mesh_data_.bounding_box_min.z() || z > mesh_data_.bounding_box_max.z()) {
            return false;
        }

        // Ray casting algorithm for point-in-polygon test
        Eigen::Vector3f point(x, y, z);
        Eigen::Vector3f direction(1.0f, 0.0f, 0.0f); // Cast ray in +X direction

        int intersections = 0;
        const size_t num_triangles = mesh_data_.indices.size() / 3;

        for (size_t i = 0; i < num_triangles; ++i) {
            size_t idx = i * 3;
            Eigen::Vector3f v0 = mesh_data_.vertices[mesh_data_.indices[idx]];
            Eigen::Vector3f v1 = mesh_data_.vertices[mesh_data_.indices[idx + 1]];
            Eigen::Vector3f v2 = mesh_data_.vertices[mesh_data_.indices[idx + 2]];

            float t;
            if (rayTriangleIntersection(point, direction, v0, v1, v2, t)) {
                if (t > 0) { // Ray intersects triangle in positive direction
                    intersections++;
                }
            }
        }

        // Odd number of intersections means point is inside
        return (intersections % 2) == 1;
    }

    std::map<std::string, double> MeshEntity::getProperties() const {
        std::map<std::string, double> props;
        props["num_vertices"] = static_cast<double>(mesh_data_.vertices.size());
        props["num_triangles"] = static_cast<double>(mesh_data_.indices.size() / 3);
        props["bounding_box_width"] = mesh_data_.bounding_box_max.x() - mesh_data_.bounding_box_min.x();
        props["bounding_box_height"] = mesh_data_.bounding_box_max.y() - mesh_data_.bounding_box_min.y();
        props["bounding_box_depth"] = mesh_data_.bounding_box_max.z() - mesh_data_.bounding_box_min.z();
        props["volume"] = props["bounding_box_width"] * props["bounding_box_height"] * props["bounding_box_depth"];
        return props;
    }

    bool MeshEntity::loadFromFile(const std::string& filename) {
        MeshLoader loader;
        auto mesh_data = loader.loadMeshData(filename);
        if (mesh_data.vertices.empty()) {
            return false;
        }

        mesh_data_ = mesh_data;
        computeBoundingBox();
        return true;
    }

    void MeshEntity::computeBoundingBox() {
        if (mesh_data_.vertices.empty()) {
            mesh_data_.bounding_box_min = Eigen::Vector3f::Zero();
            mesh_data_.bounding_box_max = Eigen::Vector3f::Zero();
            return;
        }

        mesh_data_.bounding_box_min = mesh_data_.vertices[0];
        mesh_data_.bounding_box_max = mesh_data_.vertices[0];

        for (const auto& vertex : mesh_data_.vertices) {
            mesh_data_.bounding_box_min = mesh_data_.bounding_box_min.cwiseMin(vertex);
            mesh_data_.bounding_box_max = mesh_data_.bounding_box_max.cwiseMax(vertex);
        }
    }

    bool MeshEntity::rayTriangleIntersection(const Eigen::Vector3f& origin,
        const Eigen::Vector3f& direction,
        const Eigen::Vector3f& v0,
        const Eigen::Vector3f& v1,
        const Eigen::Vector3f& v2,
        float& t) const {
        // Möller–Trumbore intersection algorithm
        const float EPSILON = 1e-6f;

        Eigen::Vector3f edge1 = v1 - v0;
        Eigen::Vector3f edge2 = v2 - v0;
        Eigen::Vector3f h = direction.cross(edge2);
        float a = edge1.dot(h);

        if (std::abs(a) < EPSILON) {
            return false; // Ray is parallel to triangle
        }

        float f = 1.0f / a;
        Eigen::Vector3f s = origin - v0;
        float u = f * s.dot(h);

        if (u < 0.0f || u > 1.0f) {
            return false;
        }

        Eigen::Vector3f q = s.cross(edge1);
        float v = f * direction.dot(q);

        if (v < 0.0f || u + v > 1.0f) {
            return false;
        }

        t = f * edge2.dot(q);
        return t > EPSILON;
    }

    // MeshLoader implementation
    MeshLoader::MeshLoader() : import_flags_(0) {
        setDefaultImportFlags();
    }

    void MeshLoader::setDefaultImportFlags() {
        import_flags_ = aiProcess_Triangulate |
            aiProcess_GenNormals |
            aiProcess_CalcTangentSpace |
            aiProcess_JoinIdenticalVertices |
            aiProcess_RemoveRedundantMaterials |
            aiProcess_FixInfacingNormals |
            aiProcess_ImproveCacheLocality |
            aiProcess_OptimizeMeshes |
            aiProcess_OptimizeGraph;
    }

    std::shared_ptr<MeshEntity> MeshLoader::loadMesh(const std::string& filename) {
        MeshData mesh_data = loadMeshData(filename);
        if (mesh_data.vertices.empty()) {
            return nullptr;
        }

        return std::make_shared<MeshEntity>(mesh_data);
    }

    MeshData MeshLoader::loadMeshData(const std::string& filename) {
        MeshData mesh_data;
        last_error_.clear();

        // Import the scene
        const aiScene* scene = importer_.ReadFile(filename, import_flags_);

        if (!scene) {
            last_error_ = "Failed to load file: " + std::string(importer_.GetErrorString());
            return mesh_data;
        }

        if (!scene->mRootNode) {
            last_error_ = "Scene has no root node";
            return mesh_data;
        }

        // Process the scene
        if (!processScene(scene, mesh_data)) {
            last_error_ = "Failed to process scene";
            return mesh_data;
        }

        mesh_data.name = filename;
        return mesh_data;
    }

    std::vector<std::string> MeshLoader::getSupportedFormats() const {
        std::vector<std::string> formats;

        aiString extension;
        for (size_t i = 0; i < importer_.GetImporterCount(); ++i) {
            const aiImporterDesc* desc = importer_.GetImporterInfo(i);
            if (desc) {
                std::string ext_list(desc->mFileExtensions);
                size_t pos = 0;
                while ((pos = ext_list.find(';')) != std::string::npos) {
                    formats.push_back("." + ext_list.substr(0, pos));
                    ext_list.erase(0, pos + 1);
                }
                if (!ext_list.empty()) {
                    formats.push_back("." + ext_list);
                }
            }
        }

        return formats;
    }

    bool MeshLoader::isFormatSupported(const std::string& extension) const {
        auto formats = getSupportedFormats();
        std::string ext = extension;
        if (ext[0] != '.') {
            ext = "." + ext;
        }

        return std::find(formats.begin(), formats.end(), ext) != formats.end();
    }

    bool MeshLoader::processScene(const aiScene* scene, MeshData& mesh_data) {
        if (!scene->mRootNode) {
            return false;
        }

        processNode(scene->mRootNode, scene, mesh_data);

        if (mesh_data.vertices.empty()) {
            return false;
        }

        // Compute bounding box
        if (!mesh_data.vertices.empty()) {
            mesh_data.bounding_box_min = mesh_data.vertices[0];
            mesh_data.bounding_box_max = mesh_data.vertices[0];

            for (const auto& vertex : mesh_data.vertices) {
                mesh_data.bounding_box_min = mesh_data.bounding_box_min.cwiseMin(vertex);
                mesh_data.bounding_box_max = mesh_data.bounding_box_max.cwiseMax(vertex);
            }
        }

        return true;
    }

    void MeshLoader::processNode(const aiNode* node, const aiScene* scene, MeshData& mesh_data) {
        // Process all meshes in this node
        for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            processMesh(mesh, mesh_data);
        }

        // Process all children
        for (unsigned int i = 0; i < node->mNumChildren; ++i) {
            processNode(node->mChildren[i], scene, mesh_data);
        }
    }

    bool MeshLoader::processMesh(const aiMesh* mesh, MeshData& mesh_data) {
        if (!mesh->HasPositions()) {
            return false;
        }

        size_t vertex_offset = mesh_data.vertices.size();

        // Process vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
            mesh_data.vertices.push_back(convertVector(mesh->mVertices[i]));
        }

        // Process normals
        if (mesh->HasNormals()) {
            for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
                mesh_data.normals.push_back(convertVector(mesh->mNormals[i]));
            }
        }

        // Process texture coordinates
        if (mesh->HasTextureCoords(0)) {
            for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
                mesh_data.tex_coords.push_back(convertVector2D(mesh->mTextureCoords[0][i]));
            }
        }

        // Process faces/indices
        for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
            aiFace face = mesh->mFaces[i];

            // Triangulate if necessary (should be handled by aiProcess_Triangulate)
            for (unsigned int j = 0; j < face.mNumIndices; ++j) {
                mesh_data.indices.push_back(vertex_offset + face.mIndices[j]);
            }
        }

        return true;
    }

    Eigen::Vector3f MeshLoader::convertVector(const aiVector3D& vec) const {
        return Eigen::Vector3f(vec.x, vec.y, vec.z);
    }

    Eigen::Vector2f MeshLoader::convertVector2D(const aiVector3D& vec) const {
        return Eigen::Vector2f(vec.x, vec.y);
    }

} // namespace voxelization
