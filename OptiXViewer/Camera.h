#pragma once

#include "glm/glm.hpp"

enum class MouseState {
	LEFT_DOWN,
	RIGHT_DOWN,
	IDLE
};

struct Camera {
	glm::vec3 eye;
	glm::vec3 cen;
	glm::vec3 up;
	float nearLen;
	float farLen;
	float cosFovy;
	// int width, height;
};

class EditMode {
private:
	EditMode() : mouseState(MouseState::IDLE), camera({ { 0,0,10 }, { 0,0,0 }, { 0,1,0 }, 0.1, 100 ,0.66f}) {}
	MouseState mouseState;
	static EditMode* editMode;
public:
	Camera camera;

	static EditMode& getEditMode();
	void setMouseMode(MouseState mouseState);
	void setCamera(Camera camera);
	void updateCamera();
	glm::mat4 getCT();
	glm::mat4 getPT(int displayW, int displayH);
	glm::mat4 getT(int displayW, int displayH);
};