#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "helper_math.h"
#include <set>

class ObjectModel {
public:
	std::vector<float3> vertices;
	std::vector<uint3> indices;
	//std::vector<glm::vec3> n_vertex;
	//std::vector<glm::vec3> n_tris;

};

std::vector<ObjectModel*> fastLoadOBJModels(const std::string& fileName);
