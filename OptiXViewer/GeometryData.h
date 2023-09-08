#pragma once
#include <cuda.h>
#include "helper_math.h"

class GeometryData {
public:
	uint3* indices;
	float3* attributes;
	size_t      numIndices;    // Count of unsigned ints, not triplets.
	size_t      numAttributes; // Count of VertexAttributes structs.
	CUdeviceptr gas;
};