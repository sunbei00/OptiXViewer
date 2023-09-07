#include "OptiXRenderer.h"
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

#define MAX_TRACE_DEPTH 2

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) NullRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

template<typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DataRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

using RaygenRecord = NullRecord;
using MissRecord = NullRecord;
using HitRecord = NullRecord;

OptiXRenderer::OptiXRenderer() {
	initOptix();
	createContext();
	createModule();
	createRaygenPrograms();
	createMissPrograms();
	createHitPrograms();
	createPipeline();
	buildSBT();
}


void OptiXRenderer::initOptix() {
	std::cout << "initializaing optix" << std::endl;

	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		throw std::runtime_error("no CUDA capable devices found!");
	std::cout << "found " << numDevices << " CUDA devices" << std::endl;

	OPTIX_CHECK(optixInit());
	std::cout << "successfully initialized optix." << std::endl;
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
	fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void OptiXRenderer::createContext() {
	const int deviceID = 0;
	CUDA_CHECK(cudaSetDevice(deviceID));
	CUDA_CHECK(cudaStreamCreate(&cudaStream));

	cudaGetDeviceProperties(&deviceProp, deviceID);
	std::cout << "running on device : " << deviceProp.name << std::endl;

	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		fprintf(stderr, "Error querying current context : error cuda %d\n", cuRes);

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

void OptiXRenderer::createModule() {
	moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions = {};
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	pipelineLinkOptions.maxTraceDepth = MAX_TRACE_DEPTH;

	std::string str;
	std::ifstream file(".\\devicePrograms.ptx", std::ios::binary);
	if (file.good())
	{
		// Found usable source file
		std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
		str.assign(buffer.begin(), buffer.end());
	}

	
	char log[2048];
	size_t sizeof_log = sizeof(log);
#if OPTIX_VERSION >= 70700
	OPTIX_CHECK(optixModuleCreate(optixContext, &moduleCompileOptions, &pipelineCompileOptions, str.c_str(), str.size(), log, &sizeof_log, &module));
#else
	OPTIX_CHECK(optixModuleCreateFromPTX(optixContext, &moduleCompileOptions, &pipelineCompileOptions, str.c_str(), str.size(), log, &sizeof_log, &module));
#endif
	if (sizeof_log > 1)
		PRINT(log);
}

void OptiXRenderer::createRaygenPrograms() {
	raygenPGs.resize(1);
	
	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = module;
	pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &raygenPGs[0]));
	if (sizeof_log > 1)
		PRINT(log);
}

void OptiXRenderer::createMissPrograms() {
	missPGs.resize(RAY_TYPE_COUNT);

	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc.miss.module = module;

	pgDesc.miss.entryFunctionName = "__miss__radiance";

	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &missPGs[RAY_TYPE_GEN]));
	if (sizeof_log > 1)
		PRINT(log);
}

void OptiXRenderer::createHitPrograms() {
	hitgroupPGs.resize(RAY_TYPE_COUNT);

	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleAH = module;
	pgDesc.hitgroup.moduleCH= module;

	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &hitgroupPGs[RAY_TYPE_GEN]));
	if (sizeof_log > 1) 
		PRINT(log);
}

void OptiXRenderer::createPipeline() {
	std::vector<OptixProgramGroup> programGroups;
	for (auto pg : raygenPGs)
		programGroups.push_back(pg);
	for (auto pg : missPGs)
		programGroups.push_back(pg);
	for (auto pg : hitgroupPGs)
		programGroups.push_back(pg);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(optixContext,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		programGroups.data(),
		(int)programGroups.size(),
		log, &sizeof_log,
		&pipeline
	));
	if (sizeof_log > 1) PRINT(log);
	
#pragma region not_use
	//const OptixStackSizes stackSizes;
	//stackSizes.cssAH;
	//
	//unsigned int maxTraceDepth = MAX_TRACE_DEPTH;
	//unsigned int maxCCDepth = MAX_TRACE_DEPTH;
	//unsigned int maxDCDepth = MAX_TRACE_DEPTH;

	//unsigned int directCallableStackSizeFromTraversal;
	//unsigned int directCallableStackSizeFromState;
	//unsigned int continuationStackSize;
	//
	//OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes,
	//	maxTraceDepth,
	//	maxCCDepth,
	//	maxDCDepth,
	//	&directCallableStackSizeFromTraversal,
	//	&directCallableStackSizeFromState,
	//	&continuationStackSize));
#pragma endregion

	OPTIX_CHECK(optixPipelineSetStackSize (pipeline, MAX_TRACE_DEPTH * 1024 * 2, MAX_TRACE_DEPTH * 1024 * 2, MAX_TRACE_DEPTH * 1024 * 2, MAX_TRACE_DEPTH));
	if (sizeof_log > 1) PRINT(log);
}

void OptiXRenderer::buildSBT() {

	std::vector<RaygenRecord> raygenRecords;
	for (int i = 0; i < raygenPGs.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		raygenRecords.push_back(rec);
	}
	rayGenRecordsBuffer.alloc_and_upload(raygenRecords);
	sbt.raygenRecord = rayGenRecordsBuffer.d_pointer();

	std::vector<MissRecord> missRecords;
	for (int i = 0; i < missPGs.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
		missRecords.push_back(rec);
	}
	missRecordBuffer.alloc_and_upload(missRecords);
	sbt.missRecordBase = missRecordBuffer.d_pointer();
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = (int)missRecords.size();


	//int numObjects = (int)model->meshes.size();
	//std::vector<HitRecord> hitgroupRecords;
	//for (int meshID = 0; meshID < numObjects; meshID++) {
	//	auto mesh = model->meshes[meshID];

	//	HitRecord rec;
	//	// hitgroupPgs[0] -> hitgroupPGs[meshID] ¼öÁ¤ÇÔ.
	//	OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[meshID], &rec));
	//	rec.data.color = mesh->diffuse;
	//	if (mesh->diffuseTextureID >= 0) {
	//		rec.data.hasTexture = true;
	//		rec.data.texture = textureObjects[mesh->diffuseTextureID];
	//	}
	//	else {
	//		rec.data.hasTexture = false;
	//	}
	//	rec.data.index = (vec3i*)indexBuffer[meshID].d_pointer();
	//	rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
	//	rec.data.normal = (vec3f*)normalBuffer[meshID].d_pointer();
	//	rec.data.texcoord = (vec2f*)texcoordBuffer[meshID].d_pointer();
	//	hitgroupRecords.push_back(rec);
	//}
	//hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	//sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
	//sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
	//sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}
