#include "OptiXRenderer.h"
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include "Camera.h"
#include "helper_math.h"


#define MAX_TRACE_DEPTH 4

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) NullRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	void* data;
};

template<typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DataRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};


using RaygenRecord = NullRecord;
using MissRecord = NullRecord;
using HitRecord = DataRecord<GeometryRecord>;

OptiXRenderer::OptiXRenderer() {
	initLaunchParams();
	initOptix();
	createContext();
	createModule();
	createRaygenPrograms();
	createMissPrograms();
	createHitPrograms();
	createPipeline();
}


void OptiXRenderer::initLaunchParams() {
	launchData.colorBuffer = nullptr;
	launchData.frameSize = make_int2(-1, -1);
	launchData.traversable = NULL;
	launchDataBuffer.alloc(sizeof(LaunchData));
	launchDataBuffer.upload(&launchData, 1);

	launchParams.data = (LaunchData*) launchDataBuffer.d_pointer();
	launchParamsBuffer.alloc(sizeof(LaunchParams));
	launchParamsBuffer.upload(&launchParams, 1);
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
	pipelineCompileOptions.usesMotionBlur = 0;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	pipelineLinkOptions.maxTraceDepth = MAX_TRACE_DEPTH;
	pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

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
	pgDesc.hitgroup.moduleCH = module;

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
		(unsigned int) programGroups.size(),
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

	OPTIX_CHECK(optixPipelineSetStackSize(pipeline, MAX_TRACE_DEPTH * 1024 * 2, MAX_TRACE_DEPTH * 1024 * 2, MAX_TRACE_DEPTH * 1024 * 2, MAX_TRACE_DEPTH));
	if (sizeof_log > 1) PRINT(log);
}

void OptiXRenderer::buildSBT() {
	std::vector<RaygenRecord> raygenRecords;
	for (int i = 0; i < raygenPGs.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		rec.data = nullptr;
		raygenRecords.push_back(rec);
	}
	printf("ray gen record : %d\n", raygenRecords.size());
	rayGenRecordsBuffer.alloc_and_upload(raygenRecords);
	sbt.raygenRecord = rayGenRecordsBuffer.d_pointer();

	std::vector<MissRecord> missRecords;
	for (int i = 0; i < missPGs.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
		rec.data = nullptr;
		missRecords.push_back(rec);
	}
	missRecordBuffer.alloc_and_upload(missRecords);
	sbt.missRecordBase = missRecordBuffer.d_pointer();
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = (unsigned int)missRecords.size();

	printf("missRecord record : %d\n", missRecords.size());

	int numObjects = (int)geoDatas.size();
	std::vector<HitRecord> hitgroupRecords;
	for (int meshID = 0; meshID < numObjects; meshID++) {
		auto& mesh = geoDatas[meshID];

		HitRecord rec;
		// hitgroupPgs[0] -> hitgroupPGs[meshID] 수정함. -> undo
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));
		//rec.data.color = mesh->diffuse;
		rec.data = GeometryRecord{ mesh.indices , mesh.attributes };
		hitgroupRecords.push_back(rec);
	}
	hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
	sbt.hitgroupRecordCount = (unsigned int)hitgroupRecords.size();

	printf("hit group record : %d\n", hitgroupRecords.size());
}


OptixTraversableHandle OptiXRenderer::createGeometryAS(ObjectModel& model) {
	auto& attributes = model.vertices;
	auto& indices = model.indices;
	CUDABuffer dAttributes;
	CUDABuffer dIndices;

	dAttributes.alloc_and_upload<float3>(attributes);
	dIndices.alloc_and_upload<uint3>(indices);

	OptixBuildInput triangleInput = {};

	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	triangleInput.triangleArray.numVertices = (unsigned int)attributes.size();
	CUdeviceptr tmp = dAttributes.d_pointer();
	triangleInput.triangleArray.vertexBuffers = &tmp;

	triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangleInput.triangleArray.indexStrideInBytes = sizeof(uint3);
	triangleInput.triangleArray.numIndexTriplets = (unsigned int) indices.size();
	triangleInput.triangleArray.indexBuffer = dIndices.d_pointer();

	unsigned int triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

	triangleInput.triangleArray.flags = triangleInputFlags;
	triangleInput.triangleArray.numSbtRecords = 1;

	OptixAccelBuildOptions accelBuildOptions = {};

	accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes accelBufferSizes;

	OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelBuildOptions, &triangleInput, (unsigned int)1, &accelBufferSizes));

	CUDABuffer tempBuffer;
	CUDABuffer outputBuffer;
	tempBuffer.alloc(accelBufferSizes.tempSizeInBytes);
	outputBuffer.alloc(accelBufferSizes.outputSizeInBytes);

	OptixTraversableHandle traversableHandle = 0;

	OPTIX_CHECK(optixAccelBuild(
		optixContext, cudaStream, &accelBuildOptions, &triangleInput, 1, tempBuffer.d_pointer(),
		tempBuffer.sizeInBytes, outputBuffer.d_pointer(), outputBuffer.sizeInBytes, &traversableHandle, nullptr, 0));

	CUDA_SYNC_CHECK();

	tempBuffer.free();

	GeometryData geometry;

	geometry.indices = (uint3*) dIndices.d_pointer();
	geometry.attributes = (float3*) dAttributes.d_pointer();
	//geometry.numIndices = indices.size() * 3;
	//geometry.numAttributes = attributes.size() * 3;
	geometry.gas = outputBuffer.d_pointer();

	geoDatas.push_back(geometry);
	geoTraversableHandle.push_back(traversableHandle);
	return traversableHandle;
}

void OptiXRenderer::createInstances() {
	for (int i = 0; i < geoTraversableHandle.size(); i++) {
		OptixInstance instance;
		// row-wise
		float tmp[12] = {	1,0,0,
							0,1,0,
							0,0,1, 
							0,0,0};
		for (int j = 0; j < 12; j++)
			instance.transform[j] = tmp[j];
		// TODO
		// instance.sbtOffset = NUM_RAY_TYPES * hitRecord;   RAY 수 늘리면 어떻게 되는지 생각해야함 
		instance.sbtOffset = 0; //i * RAY_TYPE_COUNT; //RAY_TYPE_COUNT * geoDatas.size();
		instance.visibilityMask = OptixVisibilityMask(255);
		instance.flags = OPTIX_INSTANCE_FLAG_NONE;
		instance.traversableHandle = geoTraversableHandle[i];
		instance.instanceId = i;
		instances.push_back(instance);
	}
}

OptixTraversableHandle OptiXRenderer::createInstancesAS() {
	CUDABuffer instancesBuffer;
	instancesBuffer.alloc_and_upload<OptixInstance>(instances);

	//OptixBuildInputInstanceArray instanceInput = {};
	//instanceInput.numInstances = 1;
	//instanceInput.instances = instancesBuffer.d_pointer();

	OptixBuildInput instanceInput = {};

	instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instanceInput.instanceArray.instances = instancesBuffer.d_pointer();
	instanceInput.instanceArray.numInstances = instances.size();

	OptixAccelBuildOptions accelBuildOptions = {};

	accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;// | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
	accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes accelBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelBuildOptions, &instanceInput, 1, &accelBufferSizes));

	CUDABuffer tempBuffer;
	CUDABuffer outputBuffer;
	tempBuffer.alloc(accelBufferSizes.tempSizeInBytes);
	outputBuffer.alloc(accelBufferSizes.outputSizeInBytes);

	OptixTraversableHandle traversableHandle;

	OPTIX_CHECK(optixAccelBuild(optixContext, cudaStream,
		&accelBuildOptions, &instanceInput, 1,
		tempBuffer.d_pointer(), accelBufferSizes.tempSizeInBytes,
		outputBuffer.d_pointer(), accelBufferSizes.outputSizeInBytes,
		&traversableHandle, nullptr, 0));

	CUDA_SYNC_CHECK();


	tempBuffer.free();
	instancesBuffer.free();

	insTraversableHandle = traversableHandle;

	return traversableHandle;
}

void OptiXRenderer::render(size_t width, size_t height) {
	if(launchData.frameSize.x != width || launchData.frameSize.y != height){
		launchData.frameSize.x = width;
		launchData.frameSize.y = height;
		launchData.traversable = insTraversableHandle;
		renderBuffer.resize(width * height * sizeof(uint32_t));
		launchData.colorBuffer = (uint32_t*)renderBuffer.d_pointer();
	}
	auto& camera = EditMode::getEditMode().camera;
	launchData.camera.position = make_float3(camera.eye.x, camera.eye.y, camera.eye.z);
	launchData.camera.direction = normalize(make_float3(camera.cen.x - camera.eye.x, camera.cen.y - camera.eye.y, camera.cen.z - camera.eye.z));
	float3 up = make_float3(camera.up.x, camera.up.y, camera.up.z);
	const float aspect = (float)width / (float)height;
	launchData.camera.horizontal = camera.cosFovy * aspect * normalize(cross(launchData.camera.direction, up));
	launchData.camera.vertical = camera.cosFovy * normalize(cross(launchData.camera.horizontal, launchData.camera.direction));


	launchDataBuffer.upload(&launchData, 1);

	OPTIX_CHECK(optixLaunch(
		pipeline, cudaStream,
		launchParamsBuffer.d_pointer(),
		launchParamsBuffer.sizeInBytes,
		&sbt,
		width,
		height,
		1
	));
	CUDA_SYNC_CHECK();
}

void OptiXRenderer::downloadPixels(uint32_t* h_pixels) {
	renderBuffer.download<uint32_t>(h_pixels, launchData.frameSize.x * launchData.frameSize.y);
}

void OptiXRenderer::updateInstancesAS() {

}


//void SampleRenderer::setCamera(const Camera& camera)
//{
//	lastSetCamera = camera;
//	launchParams.camera.position = camera.from;
//	launchParams.camera.direction = normalize(camera.at - camera.from);
//	const float cosFovy = 0.66f;
//	const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
//	launchParams.camera.horizontal
//		= cosFovy * aspect * normalize(cross(launchParams.camera.direction,
//			camera.up));
//	launchParams.camera.vertical
//		= cosFovy * normalize(cross(launchParams.camera.horizontal,
//			launchParams.camera.direction));
//}