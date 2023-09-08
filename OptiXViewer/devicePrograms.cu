#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include <glm/fwd.hpp>

extern "C" __constant__ LaunchParams optixLaunchParams;


static __forceinline__ __device__
void * unpackPointer(uint32_t i0, uint32_t i1 ) {
const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
void* ptr = reinterpret_cast < void *> (uptr); 
return ptr;
}

static __forceinline__ __device__
void  packPointer(void * ptr, uint32_t& i0, uint32_t& i1 ) {
const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
i0 = uptr >> 32;
i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD() { 
const uint32_t u0 = optixGetPayload_0();
const uint32_t u1 = optixGetPayload_1();
return reinterpret_cast <T *> (unpackPointer(u0, u1));
}

extern "C" __global__ void __closesthit__radiance() {
    //printf("yah!");
    GeometryRecord& sbtData = *(GeometryRecord*) optixGetSbtDataPointer();

    const int primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.indices[primID];
    const float3& A = sbtData.attributes[index.x];
    const float3& B = sbtData.attributes[index.y];
    const float3& C = sbtData.attributes[index.z];
    const float3 Ng = normalize(cross(B - A, C - A));

    const float3 rayDir = optixGetWorldRayDirection();
    const float cosDN = 0.2f + .8f * fabsf(dot(rayDir, Ng));
    float3& prd = *(float3*)getPRD<float3>();
    prd = cosDN * make_float3(0.6,0.6,0.6);
}
  
extern "C" __global__ void __anyhit__radiance() {
    // printf("test");
}

extern "C" __global__ void __miss__radiance(){
    float3& prd = *(float3*)getPRD<float3>();
    // set to constant white as background color
    prd = make_float3(1.f);
}

extern "C" __global__ void __raygen__renderFrame() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    auto& camera = optixLaunchParams.data->camera;

    float3 pixelColorPRD = make_float3(0.f,0.f,0.f);
    uint32_t u0, u1;
    packPointer(&pixelColorPRD, u0, u1);

    const float2 screen(make_float2(ix + .5f, iy + .5f)
        / make_float2(optixLaunchParams.data->frameSize));

    float3 rayDir = normalize(camera.direction
        + (screen.x - 0.5f) * camera.horizontal
        + (screen.y - 0.5f) * camera.vertical);

    optixTrace(optixLaunchParams.data->traversable,
        camera.position,
        rayDir,
        0.f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_GEN,             // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        RAY_TYPE_GEN,             // missSBTIndex 
        u0, u1);
    const int r = int(255.99f * pixelColorPRD.x);
    const int g = int(255.99f * pixelColorPRD.y);
    const int b = int(255.99f * pixelColorPRD.z);

    const uint32_t rgba = 0xff000000
        | (r << 0) | (g << 8) | (b << 16);

    const uint32_t fbIndex = ix + iy * optixLaunchParams.data->frameSize.x;
    optixLaunchParams.data->colorBuffer[fbIndex] = rgba;
}
  