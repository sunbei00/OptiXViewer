#pragma once
#include "helper_math.h"

typedef struct {
	bool isPlane;
	float gridSize = 0.3;

	int albedoID;
	float3 color;
	float3 emission;
	float metallic;
	float subsurface;
	float specular;
	float roughness;
	float specularTint;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	// BrdfType brdf;
} Matrial;