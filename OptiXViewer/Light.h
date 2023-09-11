#pragma once
#include "helper_math.h"

class Light {
public:
	float3 pos = make_float3(0,5,0);
	float3 color = make_float3(1,1,1);
	float3 uv = make_float3(1,0.2f,1);
	float3 dir = make_float3(0,-1,0);
	
};