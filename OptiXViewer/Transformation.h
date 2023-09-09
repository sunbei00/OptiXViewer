#pragma once
#include "glm/glm.hpp"
#include <glm/ext/matrix_transform.hpp>

class Transformation {
	glm::mat4 mTransfomation;
public:
	glm::vec3 mTanslation;
	glm::vec3 mRotation;
	glm::vec3 mScale;
	Transformation() : mScale(glm::vec3(1,1,1)) , mTanslation(glm::vec3(0,0,0)), mRotation(0,0,0), mTransfomation(){}
	void setRotation(float eulerX, float eulerY, float eulerZ) {
		mRotation.x = eulerX;
		mRotation.y = eulerY;
		mRotation.z = eulerZ;
	}
	void rotate(float eulerX, float eulerY, float eulerZ) {
		mRotation.x += eulerX;
		mRotation.y += eulerY;
		mRotation.z += eulerZ;
	}
	void setTranslation(float x, float y, float z) {
		mTanslation.x = x;
		mTanslation.y = y;
		mTanslation.z = z;
	}
	void Translation(float x, float y, float z) {
		mTanslation.x += x;
		mTanslation.y += y;
		mTanslation.z += z;
	}
	void setScale(float x, float y, float z) {
		mScale.x = x;
		mScale.y = y;
		mScale.z = z;
	}
	void scale(float x, float y, float z) {
		mScale.x += x;
		mScale.y += y;
		mScale.z += z;
	}
	float* getMatrix(size_t* size) {
		*size = 12;
		mTransfomation = glm::mat4(1.0f);
		glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), mScale);
		glm::mat4 rotationXMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(mRotation.x), glm::vec3(1,0,0));
		glm::mat4 rotationYMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(mRotation.y), glm::vec3(0,1,0));
		glm::mat4 rotationZMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(mRotation.z), glm::vec3(0,0,1));
		glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), mTanslation);

		mTransfomation = translationMatrix * rotationZMatrix * rotationYMatrix * rotationXMatrix * scaleMatrix * mTransfomation;
		mTransfomation = glm::transpose(mTransfomation); // row-major
		
		float* ret = new float[*size];
		memcpy(ret, &mTransfomation, *size * sizeof(float));
		return ret;
	}
};