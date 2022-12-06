/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Samuel Repka <xrepka07@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    5.12.2022
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <omp.h>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree"),
    mPerThreadTriangles(omp_get_max_threads())
{
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField& field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    #pragma omp parallel shared(field)
    {
        #pragma omp single nowait
        {
            octreeDive(Vec3_t<float>(0), mGridSize, field);
        }
    }

    // merge thread vectors
    for (int i = 0; i < mPerThreadTriangles.size(); i++) {
        auto t = mPerThreadTriangles[i];
        mTriangles.insert(mTriangles.end(), t.begin(), t.end());
    }

    return mTriangles.size();
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float>& pos, const ParametricScalarField& field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float>* pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    // #pragma omp parallel for default(none) shared(pos, pPoints, count) reduction(min : value)
    for (unsigned i = 0; i < count; ++i)
    {
        float distanceSquared = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t& triangle)
{
    int n = omp_get_thread_num();
    mPerThreadTriangles[n].push_back(triangle);
}

void TreeMeshBuilder::octreeDive(Vec3_t<float> pt, int size, const ParametricScalarField& field)
{
    {
        {
            if (size == 1) {
                buildCube(pt, field);
            }
            else {
                octreeSubdivide(pt, size, field);
            }
        }
    }
}


void TreeMeshBuilder::octreeSubdivide(Vec3_t<float> pt, int size, const ParametricScalarField& field) {
    int newsize = size / 2;
    float shift = newsize / 2.f;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++)
            {
                #pragma omp task firstprivate(i,j,k) shared(field)
                {
                    Vec3_t<float> newPoint(pt.x + (newsize * i), pt.y + (newsize * j), pt.z + (newsize * k));
                    Vec3_t<float> newPointContinuous(
                        (newPoint.x + shift) * mGridResolution,
                        (newPoint.y + shift) * mGridResolution,
                        (newPoint.z + shift) * mGridResolution
                    );

                    float Fp = evaluateFieldAt(newPointContinuous, field);
                    float t = mIsoLevel + C32 * newsize * mGridResolution;
                    if (Fp <= t) {
                        octreeDive(newPoint, newsize, field);
                    }
                }
            }
        }
    }
}
