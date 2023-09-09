#pragma once
#include "helper_math.h"

class Light {
public:
	float3 pos = make_float3(0,5,0);
	float3 color = make_float3(1,1,1);
	float3 uv = make_float3(3,3,3);
	float1 ambient = make_float1(0.2);

	bool isSpot = false;
	float3 spotDir = make_float3(0,-1,0);
	float1 angle = make_float1(60);
	
};