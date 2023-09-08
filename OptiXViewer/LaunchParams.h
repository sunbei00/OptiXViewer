#pragma once
#include "optix.h"
#include "helper_math.h"
#include <stdint.h>


enum RAY_TYPE { RAY_TYPE_GEN = 0, RAY_TYPE_COUNT };

struct dCamera {
	float3 position;
	float3 direction;
	float3 horizontal;
	float3 vertical;
};

struct LaunchData {
	uint32_t* colorBuffer = NULL;
	OptixTraversableHandle traversable = NULL;
	int2 frameSize = int2{ 0,0 };
	dCamera camera = {};
};


struct GeometryRecord {
	uint3* indices;
	float3* attributes;
};

struct LaunchParams {
	LaunchData* data;
};