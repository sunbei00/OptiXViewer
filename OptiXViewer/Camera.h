#pragma once

#include "glm/glm.hpp"

enum class MouseState {
	LEFT_DOWN,
	RIGHT_DOWN,
	IDLE
};

struct Camera {
	glm::vec3 cen;
	glm::vec3 eye;
	glm::vec3 up;
	float nearLen;
	float farLen;
	float cosFovy;
};

class EditMode {
private:
	EditMode() : mouseState(MouseState::IDLE), camera({ { -1.73, 1.636, 1.153 }, { -3.86526,2.401,3.759 }, { 0,1,0 }, 0.1, 100 ,0.66f}) {}
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