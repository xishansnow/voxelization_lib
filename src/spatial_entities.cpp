#include "spatial_entities.hpp"
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <limits>
#include <random>

namespace voxelization {

    // BoxEntity implementation
    BoxEntity::BoxEntity(double center_x, double center_y, double center_z,
        double size_x, double size_y, double size_z)
        : center_x_(center_x), center_y_(center_y), center_z_(center_z),
        size_x_(size_x), size_y_(size_y), size_z_(size_z) {
    }

    std::vector<double> BoxEntity::getBoundingBox() const {
        return {
            center_x_ - size_x_ / 2.0,
            center_y_ - size_y_ / 2.0,
            center_z_ - size_z_ / 2.0,
            center_x_ + size_x_ / 2.0,
            center_y_ + size_y_ / 2.0,
            center_z_ + size_z_ / 2.0
        };
    }

    bool BoxEntity::isPointInside(double x, double y, double z) const {
        return (x >= center_x_ - size_x_ / 2.0 && x <= center_x_ + size_x_ / 2.0 &&
            y >= center_y_ - size_y_ / 2.0 && y <= center_y_ + size_y_ / 2.0 &&
            z >= center_z_ - size_z_ / 2.0 && z <= center_z_ + size_z_ / 2.0);
    }

    std::map<std::string, double> BoxEntity::getProperties() const {
        return {
            {"center_x", center_x_},
            {"center_y", center_y_},
            {"center_z", center_z_},
            {"size_x", size_x_},
            {"size_y", size_y_},
            {"size_z", size_z_}
        };
    }

    // CylinderEntity implementation
    CylinderEntity::CylinderEntity(double center_x, double center_y, double center_z,
        double radius, double height)
        : center_x_(center_x), center_y_(center_y), center_z_(center_z),
        radius_(radius), height_(height) {
    }

    std::vector<double> CylinderEntity::getBoundingBox() const {
        return {
            center_x_ - radius_,
            center_y_ - radius_,
            center_z_ - height_ / 2.0,
            center_x_ + radius_,
            center_y_ + radius_,
            center_z_ + height_ / 2.0
        };
    }

    bool CylinderEntity::isPointInside(double x, double y, double z) const {
        double dx = x - center_x_;
        double dy = y - center_y_;
        double dz = z - center_z_;

        // Check if point is within height bounds
        if (std::abs(dz) > height_ / 2.0) {
            return false;
        }

        // Check if point is within radius
        return (dx * dx + dy * dy) <= (radius_ * radius_);
    }

    std::map<std::string, double> CylinderEntity::getProperties() const {
        return {
            {"center_x", center_x_},
            {"center_y", center_y_},
            {"center_z", center_z_},
            {"radius", radius_},
            {"height", height_}
        };
    }

    // SphereEntity implementation
    SphereEntity::SphereEntity(double center_x, double center_y, double center_z, double radius)
        : center_x_(center_x), center_y_(center_y), center_z_(center_z), radius_(radius) {
    }

    std::vector<double> SphereEntity::getBoundingBox() const {
        return {
            center_x_ - radius_,
            center_y_ - radius_,
            center_z_ - radius_,
            center_x_ + radius_,
            center_y_ + radius_,
            center_z_ + radius_
        };
    }

    bool SphereEntity::isPointInside(double x, double y, double z) const {
        double dx = x - center_x_;
        double dy = y - center_y_;
        double dz = z - center_z_;
        return (dx * dx + dy * dy + dz * dz) <= (radius_ * radius_);
    }

    std::map<std::string, double> SphereEntity::getProperties() const {
        return {
            {"center_x", center_x_},
            {"center_y", center_y_},
            {"center_z", center_z_},
            {"radius", radius_}
        };
    }

    // EllipsoidEntity implementation
    EllipsoidEntity::EllipsoidEntity(double center_x, double center_y, double center_z,
        double radius_x, double radius_y, double radius_z)
        : center_x_(center_x), center_y_(center_y), center_z_(center_z),
        radius_x_(radius_x), radius_y_(radius_y), radius_z_(radius_z) {
    }

    std::vector<double> EllipsoidEntity::getBoundingBox() const {
        return {
            center_x_ - radius_x_,
            center_y_ - radius_y_,
            center_z_ - radius_z_,
            center_x_ + radius_x_,
            center_y_ + radius_y_,
            center_z_ + radius_z_
        };
    }

    bool EllipsoidEntity::isPointInside(double x, double y, double z) const {
        double dx = (x - center_x_) / radius_x_;
        double dy = (y - center_y_) / radius_y_;
        double dz = (z - center_z_) / radius_z_;
        return (dx * dx + dy * dy + dz * dz) <= 1.0;
    }

    std::map<std::string, double> EllipsoidEntity::getProperties() const {
        return {
            {"center_x", center_x_},
            {"center_y", center_y_},
            {"center_z", center_z_},
            {"radius_x", radius_x_},
            {"radius_y", radius_y_},
            {"radius_z", radius_z_}
        };
    }

    // ConeEntity implementation
    ConeEntity::ConeEntity(double center_x, double center_y, double center_z,
        double radius, double height)
        : center_x_(center_x), center_y_(center_y), center_z_(center_z),
        radius_(radius), height_(height) {
    }

    std::vector<double> ConeEntity::getBoundingBox() const {
        return {
            center_x_ - radius_,
            center_y_ - radius_,
            center_z_ - height_ / 2.0,
            center_x_ + radius_,
            center_y_ + radius_,
            center_z_ + height_ / 2.0
        };
    }

    bool ConeEntity::isPointInside(double x, double y, double z) const {
        double dx = x - center_x_;
        double dy = y - center_y_;
        double dz = z - center_z_;

        // Check if point is within height bounds
        if (std::abs(dz) > height_ / 2.0) {
            return false;
        }

        // Calculate radius at current height
        double height_ratio = (height_ / 2.0 - std::abs(dz)) / (height_ / 2.0);
        double current_radius = radius_ * height_ratio;

        // Check if point is within current radius
        return (dx * dx + dy * dy) <= (current_radius * current_radius);
    }

    std::map<std::string, double> ConeEntity::getProperties() const {
        return {
            {"center_x", center_x_},
            {"center_y", center_y_},
            {"center_z", center_z_},
            {"radius", radius_},
            {"height", height_}
        };
    }

    // CompositeEntity implementation
    CompositeEntity::CompositeEntity(const std::vector<std::shared_ptr<SpatialEntity>>& entities)
        : entities_(entities) {
    }

    std::vector<double> CompositeEntity::getBoundingBox() const {
        if (entities_.empty()) {
            return { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        }

        auto bbox = entities_[0]->getBoundingBox();
        double min_x = bbox[0], max_x = bbox[3];
        double min_y = bbox[1], max_y = bbox[4];
        double min_z = bbox[2], max_z = bbox[5];

        for (size_t i = 1; i < entities_.size(); ++i) {
            bbox = entities_[i]->getBoundingBox();
            min_x = std::min(min_x, bbox[0]);
            max_x = std::max(max_x, bbox[3]);
            min_y = std::min(min_y, bbox[1]);
            max_y = std::max(max_y, bbox[4]);
            min_z = std::min(min_z, bbox[2]);
            max_z = std::max(max_z, bbox[5]);
        }

        return { min_x, min_y, min_z, max_x, max_y, max_z };
    }

    bool CompositeEntity::isPointInside(double x, double y, double z) const {
        for (const auto& entity : entities_) {
            if (entity->isPointInside(x, y, z)) {
                return true;
            }
        }
        return false;
    }

    void CompositeEntity::addEntity(const std::shared_ptr<SpatialEntity>& entity) {
        entities_.push_back(entity);
    }

    void CompositeEntity::removeEntity(size_t index) {
        if (index < entities_.size()) {
            entities_.erase(entities_.begin() + index);
        }
    }

    std::map<std::string, double> CompositeEntity::getProperties() const {
        return {
            {"entity_count", static_cast<double>(entities_.size())}
        };
    }

    // MeshEntity implementation
    MeshEntity::MeshEntity(const std::vector<Eigen::Vector3d>& vertices, const std::vector<std::vector<int>>& faces)
        : vertices_(vertices), faces_(faces) {
    }

    std::vector<double> MeshEntity::getBoundingBox() const {
        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double min_z = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::lowest();
        double max_y = std::numeric_limits<double>::lowest();
        double max_z = std::numeric_limits<double>::lowest();
        for (const auto& v : vertices_) {
            min_x = std::min(min_x, v.x());
            min_y = std::min(min_y, v.y());
            min_z = std::min(min_z, v.z());
            max_x = std::max(max_x, v.x());
            max_y = std::max(max_y, v.y());
            max_z = std::max(max_z, v.z());
        }
        return { min_x, min_y, min_z, max_x, max_y, max_z };
    }

    bool MeshEntity::isPointInside(double x, double y, double z) const {
        // 简单实现：判断点是否在包围盒内
        auto bbox = getBoundingBox();
        return x >= bbox[0] && x <= bbox[3] &&
            y >= bbox[1] && y <= bbox[4] &&
            z >= bbox[2] && z <= bbox[5];
    }

    std::map<std::string, double> MeshEntity::getProperties() const {
        // 返回顶点数和面数等属性
        std::map<std::string, double> props;
        props["vertex_count"] = static_cast<double>(vertices_.size());
        props["face_count"] = static_cast<double>(faces_.size());
        return props;
    }

    std::vector<std::shared_ptr<SpatialEntity>> SpatialEntityFactory::createRandomEntities(int num) {
        std::vector<std::shared_ptr<SpatialEntity>> entities;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (int i = 0; i < num; ++i) {
            float cx = dist(rng);
            float cy = dist(rng);
            float cz = dist(rng);
            float sx = 1.0f + dist(rng) * 0.1f;
            float sy = 1.0f + dist(rng) * 0.1f;
            float sz = 1.0f + dist(rng) * 0.1f;
            entities.push_back(std::make_shared<BoxEntity>(cx, cy, cz, sx, sy, sz));
        }
        return entities;
    }

} // namespace voxelization
