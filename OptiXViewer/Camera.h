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
	float near;
	float far;
};

class EditMode {
private:
	EditMode() : mouseState(MouseState::IDLE), camera({ { 0,0,20 }, { 0,0,0 }, { 0,1,0 }, 0.1, 10000 }) {}
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