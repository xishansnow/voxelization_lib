#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <Eigen/Core>

#include "spatial_entities.hpp"

#define _USE_MATH_DEFINES

class ComplexMeshGenerator {
private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<double> pos_dist_;
    std::uniform_real_distribution<double> size_dist_;

public:
    ComplexMeshGenerator(double min_pos, double max_pos, double min_size, double max_size)
        : gen_(rd_()),
        pos_dist_(min_pos, max_pos),
        size_dist_(min_size, max_size) {
    }

    // Generate a complex cube with internal structure
    void generateComplexCubeMesh(double x, double y, double z, double size,
        std::vector<Eigen::Vector3d>& vertices,
        std::vector<std::vector<int>>& faces) {
        // Base cube vertices
        vertices = {
            {x, y, z}, {x + size, y, z}, {x + size, y + size, z}, {x, y + size, z},
            {x, y, z + size}, {x + size, y, z + size}, {x + size, y + size, z + size}, {x, y + size, z + size}
        };

        // Add internal vertices for complexity
        double half_size = size / 2.0;
        vertices.push_back({ x + half_size, y + half_size, z + half_size }); // center
        vertices.push_back({ x + half_size, y, z + half_size }); // front center
        vertices.push_back({ x + half_size, y + size, z + half_size }); // back center
        vertices.push_back({ x, y + half_size, z + half_size }); // left center
        vertices.push_back({ x + size, y + half_size, z + half_size }); // right center

        // Complex face structure
        faces = {
            // Bottom faces
            {0, 1, 8}, {1, 2, 8}, {2, 3, 8}, {3, 0, 8},
            // Top faces
            {4, 7, 8}, {7, 6, 8}, {6, 5, 8}, {5, 4, 8},
            // Front faces
            {0, 4, 9}, {4, 5, 9}, {5, 1, 9}, {1, 0, 9},
            // Back faces
            {2, 6, 10}, {6, 7, 10}, {7, 3, 10}, {3, 2, 10},
            // Left faces
            {0, 3, 11}, {3, 7, 11}, {7, 4, 11}, {4, 0, 11},
            // Right faces
            {1, 5, 12}, {5, 6, 12}, {6, 2, 12}, {2, 1, 12}
        };
    }

    // Generate a pyramid mesh
    void generatePyramidMesh(double x, double y, double z, double size,
        std::vector<Eigen::Vector3d>& vertices,
        std::vector<std::vector<int>>& faces) {
        double half_size = size / 2.0;
        double height = size * 1.5;

        vertices = {
            // Base square
            {x, y, z}, {x + size, y, z}, {x + size, y + size, z}, {x, y + size, z},
            // Apex
            {x + half_size, y + half_size, z + height}
        };

        faces = {
            // Base
            {0, 2, 1}, {0, 3, 2},
            // Triangular faces
            {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}
        };
    }

    // Generate a torus-like mesh
    void generateTorusMesh(double x, double y, double z, double size,
        std::vector<Eigen::Vector3d>& vertices,
        std::vector<std::vector<int>>& faces) {
        double center_x = x + size / 2.0;
        double center_y = y + size / 2.0;
        double center_z = z + size / 2.0;
        double major_radius = size / 3.0;
        double minor_radius = size / 6.0;

        int major_segments = 8;
        int minor_segments = 6;

        // Generate vertices
        for (int i = 0; i < major_segments; ++i) {
            double major_angle = 2.0 * M_PI * i / major_segments;
            for (int j = 0; j < minor_segments; ++j) {
                double minor_angle = 2.0 * M_PI * j / minor_segments;

                double x_pos = center_x + (major_radius + minor_radius * cos(minor_angle)) * cos(major_angle);
                double y_pos = center_y + (major_radius + minor_radius * cos(minor_angle)) * sin(major_angle);
                double z_pos = center_z + minor_radius * sin(minor_angle);

                vertices.push_back({ x_pos, y_pos, z_pos });
            }
        }

        // Generate faces
        for (int i = 0; i < major_segments; ++i) {
            for (int j = 0; j < minor_segments; ++j) {
                int current = i * minor_segments + j;
                int next_major = ((i + 1) % major_segments) * minor_segments + j;
                int next_minor = i * minor_segments + ((j + 1) % minor_segments);
                int next_both = ((i + 1) % major_segments) * minor_segments + ((j + 1) % minor_segments);

                faces.push_back({ current, next_major, next_both });
                faces.push_back({ current, next_both, next_minor });
            }
        }
    }

    // Generate a star-shaped mesh
    void generateStarMesh(double x, double y, double z, double size,
        std::vector<Eigen::Vector3d>& vertices,
        std::vector<std::vector<int>>& faces) {
        double center_x = x + size / 2.0;
        double center_y = y + size / 2.0;
        double center_z = z + size / 2.0;

        // Center vertex
        vertices.push_back({ center_x, center_y, center_z });

        int num_points = 8;
        double outer_radius = size / 2.0;
        double inner_radius = size / 4.0;

        // Generate star points
        for (int i = 0; i < num_points; ++i) {
            double angle = 2.0 * M_PI * i / num_points;
            double outer_x = center_x + outer_radius * cos(angle);
            double outer_y = center_y + outer_radius * sin(angle);
            double outer_z = center_z + (i % 2 == 0 ? size / 3.0 : -size / 3.0);

            double inner_x = center_x + inner_radius * cos(angle + M_PI / num_points);
            double inner_y = center_y + inner_radius * sin(angle + M_PI / num_points);
            double inner_z = center_z;

            vertices.push_back({ outer_x, outer_y, outer_z });
            vertices.push_back({ inner_x, inner_y, inner_z });
        }

        // Generate triangular faces
        for (int i = 0; i < num_points; ++i) {
            int outer1 = 1 + i * 2;
            int inner1 = 2 + i * 2;
            int outer2 = 1 + ((i + 1) % num_points) * 2;
            int inner2 = 2 + ((i + 1) % num_points) * 2;

            // Triangles from center
            faces.push_back({ 0, outer1, inner1 });
            faces.push_back({ 0, inner1, outer2 });

            // Quad faces (as triangles)
            faces.push_back({ inner1, inner2, outer2 });
            faces.push_back({ inner1, outer2, outer1 });
        }
    }

    // Generate an irregular polyhedron
    void generateIrregularPolyhedron(double x, double y, double z, double size,
        std::vector<Eigen::Vector3d>& vertices,
        std::vector<std::vector<int>>& faces) {
        // Create an irregular polyhedron with random perturbations
        std::uniform_real_distribution<double> perturb_dist(-size * 0.1, size * 0.1);

        // Base vertices with perturbations
        vertices = {
            {x + perturb_dist(gen_), y + perturb_dist(gen_), z + perturb_dist(gen_)},
            {x + size + perturb_dist(gen_), y + perturb_dist(gen_), z + perturb_dist(gen_)},
            {x + size + perturb_dist(gen_), y + size + perturb_dist(gen_), z + perturb_dist(gen_)},
            {x + perturb_dist(gen_), y + size + perturb_dist(gen_), z + perturb_dist(gen_)},
            {x + perturb_dist(gen_), y + perturb_dist(gen_), z + size + perturb_dist(gen_)},
            {x + size + perturb_dist(gen_), y + perturb_dist(gen_), z + size + perturb_dist(gen_)},
            {x + size + perturb_dist(gen_), y + size + perturb_dist(gen_), z + size + perturb_dist(gen_)},
            {x + perturb_dist(gen_), y + size + perturb_dist(gen_), z + size + perturb_dist(gen_)}
        };

        // Add some internal vertices for complexity
        for (int i = 0; i < 4; ++i) {
            double mid_x = x + size / 2.0 + perturb_dist(gen_);
            double mid_y = y + size / 2.0 + perturb_dist(gen_);
            double mid_z = z + size / 2.0 + perturb_dist(gen_);
            vertices.push_back({ mid_x, mid_y, mid_z });
        }

        // Complex face structure
        faces = {
            // Bottom faces
            {0, 1, 8}, {1, 2, 8}, {2, 3, 8}, {3, 0, 8},
            // Top faces
            {4, 7, 9}, {7, 6, 9}, {6, 5, 9}, {5, 4, 9},
            // Side faces
            {0, 4, 10}, {4, 5, 10}, {5, 1, 10}, {1, 0, 10},
            {2, 6, 11}, {6, 7, 11}, {7, 3, 11}, {3, 2, 11},
            // Additional complex faces
            {0, 3, 7}, {0, 7, 4}, {1, 5, 6}, {1, 6, 2}
        };
    }

    void testAllMeshTypes() {
        std::cout << "=== Complex Mesh Generation Test ===" << std::endl;

        const char* mesh_names[] = {
            "Complex Cube", "Pyramid", "Torus", "Star", "Irregular Polyhedron"
        };

        for (int i = 0; i < 5; ++i) {
            std::vector<Eigen::Vector3d> vertices;
            std::vector<std::vector<int>> faces;

            double x = pos_dist_(gen_);
            double y = pos_dist_(gen_);
            double z = pos_dist_(gen_);
            double size = size_dist_(gen_);

            switch (i) {
            case 0:
                generateComplexCubeMesh(x, y, z, size, vertices, faces);
                break;
            case 1:
                generatePyramidMesh(x, y, z, size, vertices, faces);
                break;
            case 2:
                generateTorusMesh(x, y, z, size, vertices, faces);
                break;
            case 3:
                generateStarMesh(x, y, z, size, vertices, faces);
                break;
            case 4:
                generateIrregularPolyhedron(x, y, z, size, vertices, faces);
                break;
            }

            std::cout << "\n" << mesh_names[i] << ":" << std::endl;
            std::cout << "  Vertices: " << vertices.size() << std::endl;
            std::cout << "  Faces: " << faces.size() << std::endl;
            std::cout << "  Position: (" << x << ", " << y << ", " << z << ")" << std::endl;
            std::cout << "  Size: " << size << std::endl;

            // Test mesh entity creation
            try {
                auto mesh_entity = std::make_shared<voxelization::MeshEntity>(vertices, faces);
                std::cout << "  ✓ Mesh entity created successfully" << std::endl;
            }
            catch (const std::exception& e) {
                std::cout << "  ✗ Error creating mesh entity: " << e.what() << std::endl;
            }
        }
    }
};

int main() {
    ComplexMeshGenerator generator(-10.0, 10.0, 2.0, 8.0);
    generator.testAllMeshTypes();

    std::cout << "\n=== Test Completed ===" << std::endl;
    return 0;
}
