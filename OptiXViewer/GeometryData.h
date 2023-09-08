#pragma once
#include <cuda.h>

class GeometryData {
public:
	CUdeviceptr indices;
	CUdeviceptr attributes;
	size_t      numIndices;    // Count of unsigned ints, not triplets.
	size_t      numAttributes; // Count of VertexAttributes structs.
	CUdeviceptr gas;
};