#pragma once
#include "LaunchParams.h"
#include "optix/CUDABuffer.h"
#include "Model.h"
#include "GeometryData.h"
#include <optix.h>
#include <optix_stubs.h>
#include <driver_types.h>
#include <vector>



class OptiXRenderer {
private:
	void initOptix();
	void createContext();
	void createModule();
	void createRaygenPrograms();
	void createMissPrograms();
	void createHitPrograms();
	void createPipeline();

	CUcontext cudaContext;
	CUstream cudaStream;
	cudaDeviceProp deviceProp;

	OptixDeviceContext optixContext;
	
	OptixPipeline pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions pipelineLinkOptions = {};

	OptixModule module;
	OptixModuleCompileOptions moduleCompileOptions = {};

	std::vector<OptixProgramGroup> raygenPGs;
	std::vector<OptixProgramGroup> missPGs;
	std::vector<OptixProgramGroup> hitgroupPGs;
	CUDABuffer rayGenRecordsBuffer;
	CUDABuffer missRecordBuffer;
	CUDABuffer hitgroupRecordsBuffer;

	OptixShaderBindingTable sbt = {};
	std::vector<GeometryData> geoDatas;
	std::vector<OptixTraversableHandle> geoTraversableHandle;
	std::vector<OptixInstance> instances;
	OptixTraversableHandle insTraversableHandle;

	LaunchParams launchParams = {};
	CUDABuffer   launchParamsBuffer;

	CUDABuffer renderBuffer;
	CUDABuffer colorBuffer;
public:
	OptiXRenderer();
	void buildSBT();
	OptixTraversableHandle createGeometryAS(ObjectModel& model);
	OptixTraversableHandle createInstancesAS();
	void updateInstancesAS();
	void createInstances();
	void render(size_t width, size_t height);
};

