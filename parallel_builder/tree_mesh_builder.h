/**
 * @file    tree_mesh_builder.h
 *
 * @author  Samuel Repka <xrepka07@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    5.12.2022
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"

class TreeMeshBuilder: public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField& field);
    float evaluateFieldAt(const Vec3_t<float>& pos, const ParametricScalarField& field);
    void emitTriangle(const Triangle_t& triangle);
    const Triangle_t* getTrianglesArray() const { return mTriangles.data(); }

    void octreeDive(Vec3_t<float> pt, int size, const ParametricScalarField& field);
    void octreeSubdivide(Vec3_t<float> pt, int size, const ParametricScalarField& field);

    std::vector<std::vector<Triangle_t>> mPerThreadTriangles;   ///< array of arrays of triangles
    std::vector<Triangle_t> mTriangles; ///< Temporary array of triangles
    static constexpr float C32 = 0.8660254037844386; ///< sqrt(3)/2

};

#endif // TREE_MESH_BUILDER_H
