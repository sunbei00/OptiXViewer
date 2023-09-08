#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include <glm/fwd.hpp>

extern "C" __constant__ LaunchParams optixLaunchParams;


static __forceinline__ __device__
void * unpackPointer(uint32_t i0, uint32_t i1 )
{
const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
void* ptr = reinterpret_cast < void *> (uptr); 
return ptr;
}

static __forceinline__ __device__
void  packPointer(void * ptr, uint32_t& i0, uint32_t& i1 )
{
const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
i0 = uptr >> 32;
i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD(int x)
{ 
#define optixGetPayload(x) optixGetPayload_##x()
const int i = x*2;
const int j = x*2+1;
const uint32_t u0 = optixGetPayload(i);
const uint32_t u1 = optixGetPayload(j);
return reinterpret_cast <T *> (unpackPointer(u0, u1));
}

extern "C" __global__ void __closesthit__radiance() {
}
  
extern "C" __global__ void __anyhit__radiance() {
	/*! for this simple example, this will remain empty */ }

extern "C" __global__ void __miss__radiance(){
}

extern "C" __global__ void __raygen__renderFrame() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int r = (ix % 256);
    const int g = (iy % 256);
    const int b = ((ix + iy) % 256);

    // const uint32_t fbIndex = ix + iy * optixLaunchParams.frameSize.x;
    //optixLaunchParams.colorBuffer[fbIndex] = vec4{(float)r,(float)g,(float)b,1};
}
  