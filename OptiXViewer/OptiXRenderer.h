#pragma once
#include "LaunchParams.h"
#include "optix/CUDABuffer.h"
#include "GeometryData.h"
#include "Transformation.h"
#include "Material.h"
#include "Model.h"
#include <optix.h>
#include <optix_stubs.h>
#include <driver_types.h>
#include <vector>



class OptiXRenderer {
private:
	void initLaunchParams();
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

	std::vector<CUDABuffer> dMaterialList;

	LaunchParams launchParams = {};
	CUDABuffer   launchParamsBuffer;
	CUDABuffer launchDataBuffer;

	CUDABuffer renderBuffer;

	// for update IAS
	OptixAccelBufferSizes accelBufferSizes;
	CUDABuffer instancesBuffer;
	OptixBuildInput instanceInput = {};
	CUDABuffer outputBuffer;
	CUDABuffer tempBuffer;
	OptixTraversableHandle insTraversableHandle;
	OptixAccelBuildOptions accelBuildOptions = {};

public:
	OptiXRenderer();
	void buildSBT();
	void createInstances();
	void updateInstancesAS();
	void updateMaterial(size_t index);
	void render(size_t width, size_t height);
	void downloadPixels(uint32_t* h_pixels);
	OptixTraversableHandle createGeometryAS(ObjectModel& model);
	OptixTraversableHandle createInstancesAS();

	LaunchData launchData = {};
	std::vector<Transformation> transformationList;
	std::vector<Material> materialList;
};

