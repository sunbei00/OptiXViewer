#include "Model.h"
#include "tinyObjLoader.h"

int addVertex(ObjectModel* mesh, tinyobj::attrib_t& attributes, const tinyobj::index_t& idx, std::map<int, int>& knownVertices) {
	if (knownVertices.find(idx.vertex_index) != knownVertices.end())
		return knownVertices[idx.vertex_index];

	const float3* vertex_array = (const float3*)attributes.vertices.data();
	const float3* normal_array = (const float3*)attributes.normals.data();
	const float2* texcoord_array = (const float2*)attributes.texcoords.data();

	int newID = (int)mesh->vertices.size();
	knownVertices[idx.vertex_index] = newID;

	mesh->vertices.push_back(vertex_array[idx.vertex_index]);

	return newID;
}

std::vector<ObjectModel*> fastLoadOBJModels(const std::string& fileName) {
	std::vector<ObjectModel*> res;

	const std::string modelDir
		= fileName.substr(0, fileName.rfind('/') + 1);

	tinyobj::attrib_t attributes;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err = "";

	bool readOK
		= tinyobj::LoadObj(&attributes,
			&shapes,
			&materials,
			&err,
			&err,
			fileName.c_str(),
			modelDir.c_str(),
			/* triangulate */true);
	if (!readOK) {
		throw std::runtime_error("Could not read OBJ model from " + fileName + " : " + err);
	}

	if (materials.empty())
		throw std::runtime_error("could not parse materials ...");

	std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
	std::map<std::string, int>      knownTextures;

	for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
		tinyobj::shape_t& shape = shapes[shapeID];

		std::set<int> materialIDs;
		for (auto faceMatID : shape.mesh.material_ids)
			materialIDs.insert(faceMatID);

		std::map<int, int> knownVertices;

		for (int materialID : materialIDs) {
			ObjectModel* mesh = new ObjectModel;

			for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
				if (shape.mesh.material_ids[faceID] != materialID) continue;
				tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
				tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
				tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

				float3 idx = make_float3(addVertex(mesh, attributes, idx0, knownVertices),
					addVertex(mesh, attributes, idx1, knownVertices),
					addVertex(mesh, attributes, idx2, knownVertices));
				mesh->indices.push_back(make_uint3(idx.x, idx.y, idx.z));
			}

			if (mesh->vertices.empty())
				delete mesh;
			else
				res.push_back(mesh);
		}
	}

	return res;
}

ObjectModel* createBox(float3 leftRightButtom, float3 topBackFront) {
	ObjectModel* model = new ObjectModel;
	auto& attributes = model->vertices;
	auto& indices = model->indices;

	float left = leftRightButtom.x;
	float right = leftRightButtom.y;
	float bottom = leftRightButtom.z;
	float top = topBackFront.x;
	float back = topBackFront.y;
	float front = topBackFront.z;
	attributes.push_back(make_float3(left, bottom, back));
	attributes.push_back(make_float3(left, bottom, front));
	attributes.push_back(make_float3(left, top, front));
	attributes.push_back(make_float3(left, top, back));

	attributes.push_back(make_float3(right, bottom, front));
	attributes.push_back(make_float3(right, bottom, back));
	attributes.push_back(make_float3(right, top, back));
	attributes.push_back(make_float3(right, top, front));

	attributes.push_back(make_float3(right, bottom, back));
	attributes.push_back(make_float3(left, bottom, back));
	attributes.push_back(make_float3(left, top, back));
	attributes.push_back(make_float3(right, top, back));

	attributes.push_back(make_float3(left, bottom, front));
	attributes.push_back(make_float3(right, bottom, front));
	attributes.push_back(make_float3(right, top, front));
	attributes.push_back(make_float3(left, top, front));

	attributes.push_back(make_float3(left, bottom, back));
	attributes.push_back(make_float3(right, bottom, back));
	attributes.push_back(make_float3(right, bottom, front));
	attributes.push_back(make_float3(left, bottom, front));

	attributes.push_back(make_float3(left, top, front));
	attributes.push_back(make_float3(right, top, front));
	attributes.push_back(make_float3(right, top, back));
	attributes.push_back(make_float3(left, top, back));

	for (unsigned int i = 0; i < 6; ++i) {
		unsigned int idx = i * 4;
		indices.push_back(make_uint3(idx, idx + 1, idx + 2));
		indices.push_back(make_uint3(idx + 2, idx + 3, idx));
	}

	return model;
}


ObjectModel* createBox() {
	return createBox(make_float3(-1,1,-1), make_float3(1,-1,1));
}

ObjectModel* createPlane()
{
	return createBox(make_float3(-1, 1, -0.01), make_float3(0.01, -1, 1));
}


