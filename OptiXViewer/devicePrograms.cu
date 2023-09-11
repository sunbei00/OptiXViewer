#include <optix_device.h>
#include <cuda_runtime.h>
#include "PRD.h"
#include "LaunchParams.h"
#include "random.h"

#define M_PIf 3.14159265358979323846f 
#define RAY_COUNT 3
#define LIGHT_COUNT 3

extern "C" __constant__ LaunchParams optixLaunchParams;

__forceinline__ __device__ float SchlickFresnel(float u) {
	float m = clamp(1.0f - u, 0.0f, 1.0f);
	float n = m * m;
	return m * n * n; // pow(m,5)
}

__forceinline__ __device__ float GTR1(float NDotH, float a) {
	if (a >= 1.0f) return (1.0f / M_PIf);
	float a2 = a * a;
	float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
	return (a2 - 1.0f) / (M_PIf * logf(a2) * t);
}

__forceinline__ __device__ float GTR2(float NDotH, float a) {
	float a2 = a * a;
	float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
	return a2 / (M_PIf * t * t);
}

__forceinline__ __device__ float smithG_GGX(float NDotv, float alphaG) {
	float a = alphaG * alphaG;
	float b = NDotv * NDotv;
	return 1.0f / (NDotv + sqrtf(a + b - a * b));
}
__forceinline__ __host__ __device__ float powerHeuristic(const float a, const float b)
{
	const float t = a * a;
	return t / (t + b * b);
}

static __forceinline__ __device__
void* unpackPointer(uint32_t i0, uint32_t i1) {
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	void* ptr = reinterpret_cast <void*> (uptr);
	return ptr;
}

static __forceinline__ __device__
void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
	const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

__forceinline__ __device__ void sample(PRD& prd, const Material& mat, const float3& hitPos) {
	const float3 V = prd.wo;

	float3 dir;

	float probability = rnd(prd.seed);
	float diffuseRatio = 0.5f * (1.0f - mat.metallic);

	float r1 = rnd(prd.seed);
	float r2 = rnd(prd.seed);


	if (probability < diffuseRatio) // sample diffuse
	{
		const float r = sqrtf(r1);
		const float phi = 2.0f * M_PIf * r2;
		dir.x = r * cosf(phi);
		dir.y = r * sinf(phi);
		dir.z = sqrtf(fmaxf(0.0f, 1.0f - dir.x * dir.x - dir.y * dir.y));
	}
	else {
		float a = max(0.001f, mat.roughness);

		float phi = r1 * 2.0f * M_PIf;

		float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
		float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
		float sinPhi = sinf(phi);
		float cosPhi = cosf(phi);

		float3 half = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);

		dir = 2.0f * dot(V, half) * half - V; //reflection vector

	}
	prd.bsdfDir = dir;
	prd.origin = hitPos + 0.01f * dir;
}

__forceinline__ __device__ void pdf(PRD& prd, const Material& mat, const float3& normal) {
	float3 V = prd.wo;
	float3 L = prd.bsdfDir;

	float specularAlpha = max(0.001f, mat.roughness);
	float clearcoatAlpha = lerp(0.1f, 0.001f, mat.clearcoatGloss);

	float diffuseRatio = 0.5f * (1.f - mat.metallic);
	float specularRatio = 1.f - diffuseRatio;

	float3 half = normalize(L + V);

	float cosTheta = fabs(dot(half, normal));
	float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
	float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

	// calculate diffuse and specular pdfs and mix ratio
	float ratio = 1.0f / (1.0f + mat.clearcoat);
	float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * fabs(dot(L, half)));
	float pdfDiff = fabs(dot(L, normal)) * (1.0f / M_PIf);

	// weight pdfs according to ratios
	prd.pdf = diffuseRatio * pdfDiff + specularRatio * pdfSpec;
}

__forceinline__ __device__ float3 eval(PRD& prd, const Material& mat, const float3& N) {
	const float3& V = prd.wo;
	const float3& L = prd.bsdfDir;

	float NDotL = dot(N, L);
	float NDotV = dot(N, V);
	//if (NDotL <= 0.0f || NDotV <= 0.0f) 
	//    return make_float3(0.0f);

	const float3 H = normalize(L + V);
	float NDotH = dot(N, H);
	float LDotH = dot(L, H);

	float3 Cdlin = mat.color;
	float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.

	float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
	float3 Cspec0 = lerp(mat.specular * 0.08f * lerp(make_float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);
	float3 Csheen = lerp(make_float3(1.0f), Ctint, mat.sheenTint);

	// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
	// and mix in diffuse retro-reflection based on roughness
	float FL = SchlickFresnel(NDotL);
	float FV = SchlickFresnel(NDotV);
	float Fd90 = 0.5f + 2.0f * LDotH * LDotH * mat.roughness;
	float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

	// Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
	// 1.25 scale is used to (roughly) preserve albedo
	// Fss90 used to "flatten" retroreflection based on roughness
	float Fss90 = LDotH * LDotH * mat.roughness;
	float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
	float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

	//// specular
	//float aspect = sqrt(1-mat.anisotropic*.9);
	//float ax = max(.001f, sqrt(mat.roughness)/aspect);
	//float ay = max(.001f, sqrt(mat.roughness)*aspect);
	//float Ds = GTR2(NDotH, dot(H, X), dot(H, Y), ax, ay);

	float a = max(0.001f, mat.roughness);
	float Ds = GTR2(NDotH, a);
	float FH = SchlickFresnel(LDotH);
	float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
	float roughg = sqrt(mat.roughness * 0.5f + 0.5f);
	float Gs = smithG_GGX(NDotL, roughg) * smithG_GGX(NDotV, roughg);

	// sheen
	float3 Fsheen = FH * mat.sheen * Csheen;

	// clearcoat (ior = 1.5 -> F0 = 0.04)
	float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatGloss));
	float Fr = lerp(0.04f, 1.0f, FH);
	float Gr = smithG_GGX(NDotL, 0.25f) * smithG_GGX(NDotV, 0.25f);

	float3 out = ((1.0f / M_PIf) * lerp(Fd, ss, mat.subsurface) * Cdlin + Fsheen)
		* (1.0f - mat.metallic)
		+ Gs * Fs * Ds + 0.25f * mat.clearcoat * Gr * Fr * Dr;

	return out * clamp(dot(N, L), 0.0f, 1.0f);
}

__forceinline__ __device__ float3 light(PRD& prd, const Material& mat, const float3& hitPos, const float3& N, const float3& Ng) {

	float3 ret = make_float3(0.0f);

	auto& data = *optixLaunchParams.data;

	const float3 surfacePos = hitPos;
	const float3 surfaceNormal = N;

	float pdfInfo = 0.f;

	for (int i = 0; i < LIGHT_COUNT; i++) {
		PRD copyPRD = prd;
		float3 lightPos = data.light.pos + make_float3(data.light.uv.x * rnd(prd.seed), data.light.uv.y * rnd(prd.seed), data.light.uv.z * rnd(prd.seed));
		float3 lightDir = lightPos - surfacePos;
		float lightDist = length(lightDir);
		float lightDistSq = lightDist * lightDist;
		lightDir /= sqrtf(lightDistSq);

		if (dot(lightDir, surfaceNormal) <= 0.0f | dot(lightDir, data.light.dir) >= 0.0f)
			continue;


		bool isShadow = false;
		uint32_t u0, u1;
		packPointer(&isShadow, u0, u1);

		optixTrace(data.traversable,
			surfacePos + 0.01 * lightDir,
			lightDir,
			0.f,    // tmin
			1e20f,  // tmax
			0.0f,   // rayTime
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
			RAY_TYPE_SHADOW,               // SBT offset
			RAY_TYPE_COUNT,				   // SBT stride
			RAY_TYPE_SHADOW,               // missSBTIndex 
			u0, u1);

		if (!isShadow)
		{
			float NdotL = dot(data.light.dir, -lightDir);
			float lightPdf = lightDistSq / (data.light.dir.x * data.light.dir.y * data.light.dir.z * NdotL);

			copyPRD.bsdfDir = lightDir;
			pdf(copyPRD, mat, N);
			float3 f = eval(copyPRD, mat, N);
			pdfInfo += copyPRD.pdf;

			ret += powerHeuristic(lightPdf, prd.pdf) * prd.throughput * f * data.light.color / max(0.001f, lightPdf);
		}
	}

	ret /= LIGHT_COUNT;
	prd.pdf = pdfInfo / LIGHT_COUNT;

	return ret;
}

template<typename T>
static __forceinline__ __device__ T* getPRD() {
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast <T*> (unpackPointer(u0, u1));
}

extern "C" __global__ void __closesthit__radiance() {
	const GeometryRecord& sbtData = *(GeometryRecord*)optixGetSbtDataPointer();
	const Material& mat = *sbtData.material;
	PRD& prd = *(PRD*)getPRD<PRD>();

	const float3 rayDir = optixGetWorldRayDirection();

	const int primID = optixGetPrimitiveIndex();
	const uint3 index = sbtData.indices[primID];
	const float3& A = sbtData.attributes[index.x];
	const float3& B = sbtData.attributes[index.y];
	const float3& C = sbtData.attributes[index.z];

	const float3 Ng = normalize(cross(B - A, C - A));
	const float3 N = faceforward(Ng, -1 * rayDir, -1 * Ng);

	const float2 theBarycentrics = optixGetTriangleBarycentrics(); // beta and gamma
	const float  alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;
	const float3 hitPos = A * alpha + B * theBarycentrics.x + C * theBarycentrics.y;
	prd.wo = -1 * rayDir;

	if (mat.isPlane && (hitPos.x <= mat.gridSize || hitPos.z <= mat.gridSize )) {
		prd.color = make_float3(0.f);
		prd.done = true;
	}
	else {
		prd.color += mat.color * prd.throughput;
		prd.color += light(prd, mat, hitPos, N, Ng);
	}


	sample(prd, mat, hitPos);
	pdf(prd, mat, N);
	float3 f = eval(prd, mat, N);

	if (prd.pdf > 0.0f)
		prd.throughput *= f / prd.pdf;
	else
		prd.done = true;
}

extern "C" __global__ void __anyhit__radiance() {
}

extern "C" __global__ void __miss__radiance() {
	PRD& prd = *(PRD*)getPRD<PRD>();
	prd.done = true;
}

extern "C" __global__ void __closesthit__shadow() {
	bool& isShadow = *(bool*)getPRD<bool>();
	isShadow = true;
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



	float3 color = make_float3(0, 0, 0);
	for (int i = 0; i < RAY_COUNT; i++) {

		const float2 screen(make_float2(ix + rnd(seed), iy + rnd(seed))
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
		prd.throughput = make_float3(1.f);
		prd.bsdfDir = rayDir;
		prd.origin = camera.position;

		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		while (true) {
			prd.wo = -prd.bsdfDir;
			optixTrace(data.traversable,
				prd.origin,
				prd.bsdfDir,
				0.f,    // tmin
				1e20f,  // tmax
				0.0f,   // rayTime
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT,        //OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
				RAY_TYPE_GEN,               // SBT offset
				RAY_TYPE_COUNT,             // SBT stride
				RAY_TYPE_GEN,               // missSBTIndex 
				u0, u1);

			if (prd.done || prd.depth >= data.maxTrace)
				break;

			prd.depth++;
		}
		color += prd.color;
	}

	color /= RAY_COUNT;


	const int r = int(255.99f * color.x);
	const int g = int(255.99f * color.y);
	const int b = int(255.99f * color.z);

	const uint32_t rgba = 0xff000000
		| (r << 0) | (g << 8) | (b << 16);

	const uint32_t fbIndex = ix + iy * data.frameSize.x;
	data.colorBuffer[fbIndex] = rgba;
}
