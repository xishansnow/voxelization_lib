#include "mesh_loader.hpp"
#include <iostream>
#include <algorithm>
#include <limits>

namespace voxelization {

    MeshLoader::MeshLoader() {
        setDefaultImportFlags();
    }

    void MeshLoader::setDefaultImportFlags() {
        import_flags_ = aiProcess_Triangulate |
            aiProcess_GenNormals |
            aiProcess_CalcTangentSpace |
            aiProcess_JoinIdenticalVertices |
            aiProcess_SortByPType;
    }

    std::shared_ptr<MeshEntity> MeshLoader::loadMesh(const std::string& filename) {
        MeshData mesh_data = loadMeshData(filename);
        if (mesh_data.vertices.empty()) {
            return nullptr;
        }

        // Convert MeshData to vertices and faces for MeshEntity constructor
        std::vector<Eigen::Vector3d> vertices;
        std::vector<std::vector<int>> faces;

        // Convert vertices from Vector3f to Vector3d
        for (const auto& v : mesh_data.vertices) {
            vertices.emplace_back(v.x(), v.y(), v.z());
        }

        // Convert indices to faces (assuming triangular faces)
        for (size_t i = 0; i < mesh_data.indices.size(); i += 3) {
            if (i + 2 < mesh_data.indices.size()) {
                std::vector<int> face = {
                    static_cast<int>(mesh_data.indices[i]),
                    static_cast<int>(mesh_data.indices[i + 1]),
                    static_cast<int>(mesh_data.indices[i + 2])
                };
                faces.push_back(face);
            }
        }

        return std::make_shared<MeshEntity>(vertices, faces);
    }

    MeshData MeshLoader::loadMeshData(const std::string& filename) {
        MeshData mesh_data;
        last_error_.clear();

        const aiScene* scene = importer_.ReadFile(filename, import_flags_);
        if (!scene) {
            last_error_ = "Failed to load mesh file: " + std::string(importer_.GetErrorString());
            return mesh_data;
        }

        if (!processScene(scene, mesh_data)) {
            last_error_ = "Failed to process mesh scene";
            return mesh_data;
        }

        return mesh_data;
    }

    bool MeshLoader::processScene(const aiScene* scene, MeshData& mesh_data) {
        if (!scene->mRootNode) {
            return false;
        }

        mesh_data.name = scene->mRootNode->mName.C_Str();
        processNode(scene->mRootNode, scene, mesh_data);

        if (mesh_data.vertices.empty()) {
            return false;
        }

        return true;
    }

    void MeshLoader::processNode(const aiNode* node, const aiScene* scene, MeshData& mesh_data) {
        // Process all meshes in this node
        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            processMesh(mesh, mesh_data);
        }

        // Process children
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNode(node->mChildren[i], scene, mesh_data);
        }
    }

    bool MeshLoader::processMesh(const aiMesh* mesh, MeshData& mesh_data) {
        if (!mesh) {
            return false;
        }

        // Process vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            Eigen::Vector3f vertex = convertVector(mesh->mVertices[i]);
            mesh_data.vertices.push_back(vertex);

            if (mesh->mNormals) {
                Eigen::Vector3f normal = convertVector(mesh->mNormals[i]);
                mesh_data.normals.push_back(normal);
            }

            if (mesh->mTextureCoords[0]) {
                Eigen::Vector2f tex_coord = convertVector2D(mesh->mTextureCoords[0][i]);
                mesh_data.tex_coords.push_back(tex_coord);
            }
        }

        // Process faces/indices
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++) {
                mesh_data.indices.push_back(face.mIndices[j]);
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

    std::vector<std::string> MeshLoader::getSupportedFormats() const {
        std::vector<std::string> formats;
        aiString extension;
        importer_.GetExtensionList(extension);

        std::string ext_str = extension.C_Str();
        size_t pos = 0;
        while ((pos = ext_str.find(';')) != std::string::npos) {
            std::string format = ext_str.substr(0, pos);
            if (!format.empty() && format[0] == '*') {
                format = format.substr(1);
            }
            formats.push_back(format);
            ext_str.erase(0, pos + 1);
        }
        if (!ext_str.empty()) {
            if (!ext_str.empty() && ext_str[0] == '*') {
                ext_str = ext_str.substr(1);
            }
            formats.push_back(ext_str);
        }

        return formats;
    }

    bool MeshLoader::isFormatSupported(const std::string& extension) const {
        auto formats = getSupportedFormats();
        return std::find(formats.begin(), formats.end(), extension) != formats.end();
    }

} // namespace voxelization
