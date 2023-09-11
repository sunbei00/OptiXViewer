#pragma once
#include "helper_math.h"

class Material{
public:
	bool isPlane = false;
	float gridSize = 0.3;

	// int albedoID = 1;
	float3 color = make_float3(0.6,0.5,0.8);
	// float3 emission = make_float3(0.6,0.6,0.6);
	float metallic = 0.210;
	float subsurface = 0.320;
	float specular = 1;
	float roughness = 0.26;
	float specularTint = 0.5;
	// float anisotropic = 0.5;
	float sheen = 1;
	float sheenTint = 1;
	float clearcoat = 1;
	float clearcoatGloss = 1;
	// float troughtput = 1.0;
};