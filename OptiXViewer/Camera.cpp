#include "Camera.h"
#include "glm/glm.hpp"
#include "imgui/imgui.h"
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/vec3.hpp>


EditMode* EditMode::editMode = nullptr;

EditMode& EditMode::getEditMode() {
	if (editMode == nullptr)
		editMode = new EditMode();
	return *editMode;
}

void EditMode::setMouseMode(MouseState mouseState)
{
	this->mouseState = mouseState;
}

void EditMode::setCamera(Camera camera)
{
	this->camera.eye = camera.eye;
	this->camera.cen = camera.cen;
	this->camera.up = camera.up;
	this->camera.near = camera.near;
	this->camera.far = camera.far;
}

void EditMode::updateCamera()
{
	static ImVec2 mousePos;
	static float alpha = 0.96f; // wheel 속도 조정
	static ImGuiIO& io = ImGui::GetIO();

	if (io.KeyCtrl &&  (io.MouseWheel) >= 0.1f) {
		camera.eye.x = camera.eye.x * 0.9 + camera.cen.x * 0.1;
		camera.eye.y = camera.eye.y * 0.9 + camera.cen.y * 0.1;
		camera.eye.z = camera.eye.z * 0.9 + camera.cen.z * 0.1;
		return;
	}
	if (glm::abs(io.MouseWheel) >= 0.1f) {
		printf("%f", io.MouseWheel);
		glm::vec3 move = io.MouseWheel * (camera.cen - camera.eye);
		camera.eye = camera.eye + move;
		camera.cen = camera.cen + move;
	}


	if (ImGui::IsMouseDragging(0, 0.5f) || ImGui::IsMouseDragging(1, 0.5f))
	{

		if (ImGui::IsMouseDown(ImGuiMouseButton_Left) || ImGui::IsMouseDown(ImGuiMouseButton_Right))
		{
			if (mouseState == MouseState::IDLE) {
				mouseState = ImGui::IsMouseDown(ImGuiMouseButton_Left) ? MouseState::LEFT_DOWN : MouseState::RIGHT_DOWN;
				mousePos = ImGui::GetMousePos();
				return;
			}
			ImVec2 curr = ImGui::GetMousePos();
			const float speed = (float)0.0006 * glm::length(camera.eye - camera.cen); // 좌클릭 이동 속도 조절
			const float rot_speed = (float)0.001 * glm::length(camera.eye - camera.cen); // 우클릭 회전 속도 조절

			if (mouseState == MouseState::LEFT_DOWN) {
				camera.eye += speed * (glm::normalize(glm::cross(camera.cen - camera.eye, camera.up)) * (mousePos.x - curr.x) + glm::normalize(camera.up) * (curr.y - mousePos.y));
				camera.cen += speed * (glm::normalize(glm::cross(camera.cen - camera.eye, camera.up)) * (mousePos.x - curr.x) + glm::normalize(camera.up) * (curr.y - mousePos.y));
			}
			if (mouseState == MouseState::RIGHT_DOWN)
				camera.cen += rot_speed * (glm::normalize(glm::cross(camera.cen - camera.eye, camera.up)) * (curr.x - mousePos.x) + glm::normalize(camera.up) * (mousePos.y - curr.y));

			mousePos = ImGui::GetMousePos();
		}
	}
	else
		mouseState = MouseState::IDLE;


}

glm::mat4 EditMode::getCT()
{
	return glm::lookAt(camera.eye, camera.cen, camera.up);
}

glm::mat4 EditMode::getPT(int displayW, int displayH)
{
	return glm::perspective(glm::radians(60.0f), (float)displayW / (float)displayH, camera.near, camera.far);
}

glm::mat4 EditMode::getT(int displayW, int displayH)
{
	return glm::perspective(glm::radians(60.0f), (float)displayW / (float)displayH, camera.near, camera.far) * glm::lookAt(camera.eye, camera.cen, camera.up);
}