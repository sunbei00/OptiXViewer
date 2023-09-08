#pragma once
#include "optix.h"

enum RAY_TYPE { RAY_TYPE_GEN, RAY_TYPE_COUNT };

struct ivec2 {
    size_t x, y;
};

struct vec4 {
    float x, y, z, w;
};

struct vec3 {
    float x, y, z;
};

struct LaunchParams {
    OptixTraversableHandle traversable = NULL;
    vec4* colorBuffer = nullptr;
    //ivec2 frameSize = ivec2{ 0,0 };

    //struct {
    //    vec3 position;
    //    vec3 direction;
    //    vec3 horizontal;
    //    vec3 vertical;
    //} camera;
};