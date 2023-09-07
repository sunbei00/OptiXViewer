#pragma once
#include "LaunchParams.h"
#include "optix/CUDABuffer.h"
#include <cuda.h>
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
	void buildSBT();

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

	LaunchParams launchParams;
	
public:
	OptiXRenderer();


};

