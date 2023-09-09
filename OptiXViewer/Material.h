#pragma once
#include "helper_math.h"

class Material{
public:
	bool isPlane = false;
	float gridSize = 0.3;

	int albedoID = 1;
	float3 color = make_float3(0.6,0.6,0.6);
	float3 emission = make_float3(0.6,0.6,0.6);
	float metallic = 0.5;
	float subsurface = 0.5;
	float specular = 0.5;
	float roughness = 0.5;
	float specularTint = 0.5;
	float anisotropic = 0.5;
	float sheen = 0.5;
	float sheenTint = 0.5;
	float clearcoat = 0.5;
	float clearcoatGloss = 0.5;
	float troughtput = 1.0;
	// BrdfType brdf;
};