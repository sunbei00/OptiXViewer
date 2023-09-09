#pragma once
#include "helper_math.h"

class PRD{
public:
	float3 color;

	int depth;
	unsigned int seed;

	// shading state
	bool done;
	bool inShadow;
	bool specularBounce;
	float3 radiance;
	float3 origin;
	float3 bsdfDir;
	float3 wo;
	float1 throughput;
	float pdf;

};