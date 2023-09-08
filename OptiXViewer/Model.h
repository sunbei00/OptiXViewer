#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "glm/glm.hpp"
#include <set>

class ObjectModel {
public:
	std::vector<glm::vec3> vertices;
	std::vector<glm::ivec3> indices;
	//std::vector<glm::vec3> n_vertex;
	//std::vector<glm::vec3> n_tris;

};

std::vector<ObjectModel*> fastLoadOBJModels(const std::string& fileName);
