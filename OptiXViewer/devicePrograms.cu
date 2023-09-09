#include <optix_device.h>
#include <cuda_runtime.h>
#include "PRD.h"
#include "LaunchParams.h"
#include "random.h"

#define M_PIf 3.14159265358979323846f 

extern "C" __constant__ LaunchParams optixLaunchParams;

inline __device__ float SchlickFresnel(float u) {
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float n = m * m;
    return m*m*n; // pow(m,5)
}

inline __device__ float GTR1(float NDotH, float a) {
    if (a >= 1.0f) return (1.0f / M_PIf);
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    return (a2 - 1.0f) / (M_PIf * logf(a2) * t);
}

inline __device__ float GTR2(float NDotH, float a) {
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    return a2 / (M_PIf * t * t);
}

inline __device__ float smithG_GGX(float NDotv, float alphaG) {
    float a = alphaG * alphaG;
    float b = NDotv * NDotv;
    return 1.0f / (NDotv + sqrtf(a + b - a * b));
}

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
    GeometryRecord& sbtData = *(GeometryRecord*) optixGetSbtDataPointer();
    Material& mat = *sbtData.material;
    auto& data = *optixLaunchParams.data;
    PRD& prd = *(PRD*)getPRD<PRD>();

    const float3 rayDir = optixGetWorldRayDirection();
    const int primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.indices[primID];
    const float3& A = sbtData.attributes[index.x];
    const float3& B = sbtData.attributes[index.y];
    const float3& C = sbtData.attributes[index.z];

    const float3 Ng = normalize(cross(B - A, C - A));
    const float3 N = faceforward(Ng, -1 * rayDir, Ng);
    //float3& V = prd.wo;
    //float3& L = prd.bsdfDir;

    //float NDotL = dot(N, L);
    //float NDotV = dot(N, V);
    //float3 H = normalize(L + V);
    //float NDotH = dot(N, H);
    //float LDotH = dot(L, H);

    //float3 Cdlin = mat.color;
    //float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.

    //float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
    //float3 Cspec0 = lerp(mat.specular * 0.08f * lerp(make_float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);
    //float3 Csheen = lerp(make_float3(1.0f), Ctint, mat.sheenTint);

    //// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    //// and mix in diffuse retro-reflection based on roughness
    //float FL = SchlickFresnel(NDotL); 
    //float FV = SchlickFresnel(NDotV);
    //float Fd90 = 0.5f + 2.0f * LDotH * LDotH * mat.roughness;
    //float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

    //// Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
    //// 1.25 scale is used to (roughly) preserve albedo
    //// Fss90 used to "flatten" retroreflection based on roughness
    //float Fss90 = LDotH * LDotH * mat.roughness;
    //float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
    //float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

    //// specular
    ////float aspect = sqrt(1-mat.anisotrokPic*.9);
    ////float ax = Max(.001f, sqr(mat.roughness)/aspect);
    ////float ay = Max(.001f, sqr(mat.roughness)*aspect);
    ////float Ds = GTR2_aniso(NDotH, Dot(H, X), Dot(H, Y), ax, ay);

    //float a = max(0.001f, mat.roughness);
    //float Ds = GTR2(NDotH, a);
    //float FH = SchlickFresnel(LDotH);
    //float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
    //float roughg = sqrt(mat.roughness * 0.5f + 0.5f);
    //float Gs = smithG_GGX(NDotL, roughg) * smithG_GGX(NDotV, roughg);

    //// sheen
    //float3 Fsheen = FH * mat.sheen * Csheen;

    //// clearcoat (ior = 1.5 -> F0 = 0.04)
    //float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatGloss));
    //float Fr = lerp(0.04f, 1.0f, FH);
    //float Gr = smithG_GGX(NDotL, 0.25f) * smithG_GGX(NDotV, 0.25f);

    //float3 out = ((1.0f / M_PIf) * lerp(Fd, ss, mat.subsurface) * Cdlin + Fsheen)
    //    * (1.0f - mat.metallic)
    //    + Gs * Fs * Ds + 0.25f * mat.clearcoat * Gr * Fr * Dr;

    //prd.color = out * clamp(dot(N, L), 0.0f, 1.0f);


    prd.color = data.light.ambient.x * sbtData.material->color;
    

    prd.done = true;
}
  
extern "C" __global__ void __anyhit__radiance() {
}

extern "C" __global__ void __miss__radiance(){
    PRD& prd = *(PRD*)getPRD<PRD>();
    prd.done = true;
}

extern "C" __global__ void __closesthit__shadow() {
}

extern "C" __global__ void __anyhit__shadow() {
}

extern "C" __global__ void __miss__shadow() {
    PRD& prd = *(PRD*)getPRD<PRD>();
    prd.done = true;
}

extern "C" __global__ void __raygen__renderFrame() {
    auto& data = *optixLaunchParams.data;
    auto& camera = optixLaunchParams.data->camera;

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    unsigned int seed = tea<16>(data.frameSize.x * iy + ix, data.frame);

    const float2 screen(make_float2(ix + .5f, iy + .5f)
        / make_float2(data.frameSize));

    float3 rayDir = normalize(camera.direction
        + (screen.x - 0.5f) * camera.horizontal
        + (screen.y - 0.5f) * camera.vertical);


    PRD prd;
    prd.color = make_float3(0);
    prd.depth = 0;
    prd.seed = seed;
    prd.done = false;
    prd.pdf = 0.0f;
    prd.specularBounce = false;
    prd.throughput = make_float1(1.f);
    prd.bsdfDir = rayDir;
    prd.origin = camera.position;

    uint32_t u0, u1;
    packPointer(&prd, u0, u1);

    while (true) {
        prd.wo = -rayDir;
        optixTrace(data.traversable,
            prd.origin,
            prd.bsdfDir,
            0.f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,        //OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_GEN,               // SBT offset
            RAY_TYPE_COUNT,             // SBT stride
            RAY_TYPE_GEN,               // missSBTIndex 
            u0, u1);

        if (prd.done || prd.depth >= data.maxTrace)
            break;

        prd.depth++;

        //ray_origin = prd.origin;
        //ray_direction = prd.bsdfDir;
    }


    const int r = int(255.99f * prd.color.x);
    const int g = int(255.99f * prd.color.y);
    const int b = int(255.99f * prd.color.z);

    const uint32_t rgba = 0xff000000
        | (r << 0) | (g << 8) | (b << 16);

    const uint32_t fbIndex = ix + iy * data.frameSize.x;
    data.colorBuffer[fbIndex] = rgba;
}
  