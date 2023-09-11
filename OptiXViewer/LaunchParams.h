#pragma once
#include "optix.h"
#include "helper_math.h"
#include "Material.h"
#include "Light.h"
#include <stdint.h>


enum RAY_TYPE { RAY_TYPE_GEN = 0 , RAY_TYPE_SHADOW  , RAY_TYPE_COUNT };

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
	Light light;
	int frame = 0;
	int maxTrace = 5;
};


struct GeometryRecord {
	uint3* indices;
	float3* attributes;
	Material* material;
};

struct LaunchParams {
	LaunchData* data;
};