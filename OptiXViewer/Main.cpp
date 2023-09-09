#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "GL/glew.h"
#include "Camera.h"
#include "GL/glu.h"
#include "OptiXRenderer.h"
#include "Model.h"
#include <sstream>
#include <stdio.h>
#include <GLFW/glfw3.h>
#include <fstream>

glm::vec3 __pair[2];
GLuint textureID = -1;
//glBegin(GL_LINES);
//glColor3f(1, 1, 1);
//glVertex3f(__pair[0].x, __pair[0].y, __pair[0].z);
//glVertex3f(__pair[1].x, __pair[1].y, __pair[1].z);
//glEnd();

static void glfw_error_callback(int error, const char* description) {
	fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

#define PLANE_ENV

int main(int, char**) {
	OptiXRenderer optixRenderer;

	//Env
	{
#ifdef PLANE_ENV
		auto box = createPlane();
		optixRenderer.createGeometryAS(*box);
#endif
#ifdef  BOX_ENV
		optixRenderer.createGeometryAS(*box);
		optixRenderer.createGeometryAS(*box);
		optixRenderer.createGeometryAS(*box);
		optixRenderer.createGeometryAS(*box);
#endif 

	}

	auto objList = fastLoadOBJModels("../../penedepthSitu/bunny_case_03_oversampled.obj");
	for (int i = 0; i < objList.size(); i++)
		optixRenderer.createGeometryAS(*objList[i]);

	optixRenderer.createInstances();
	optixRenderer.createInstancesAS();
	optixRenderer.buildSBT();

	// Env Setting
	{
#ifdef PLANE_ENV
		optixRenderer.transformationList[0].setTranslation(0, 0, 0);
		optixRenderer.transformationList[0].setRotation(0, 0, 0);
		optixRenderer.transformationList[0].setScale(10, 1, 10);
#endif
#ifdef BOX_ENV
		optixRenderer.transformationList[0].setTranslation(0, 0, 0);
		optixRenderer.transformationList[0].setRotation(0, 0, 0);
		optixRenderer.transformationList[0].setScale(4, 1, 6);

		optixRenderer.transformationList[1].setTranslation(0, 3, -6);
		optixRenderer.transformationList[1].setRotation(90, 0, 0);
		optixRenderer.transformationList[1].setScale(4, 1, 3);

		optixRenderer.transformationList[2].setTranslation(3, 3, 0);
		optixRenderer.transformationList[2].setRotation(0, 0, 90);
		optixRenderer.transformationList[2].setScale(3, 1, 6);

		optixRenderer.transformationList[3].setTranslation(0, 6, 0);
		optixRenderer.transformationList[3].setRotation(0, 0, 0);
		optixRenderer.transformationList[3].setScale(4, 1, 6);


		optixRenderer.transformationList[4].setTranslation(0, 3, 6);
		optixRenderer.transformationList[4].setRotation(90, 0, 0);
		optixRenderer.transformationList[4].setScale(4, 1, 3);
#endif
		optixRenderer.updateInstancesAS();
	}

#pragma region OTHER_INFO
	//glm::vec3 bmin(1e7f,1e7f,1e7f), bmax(-1e7f,-1e7f,-1e7f);
	//for (auto obj : objList) {
	//    for (auto v : obj->vertices) {
	//        bmin.x = min(bmin.x, v.x);
	//        bmin.y = min(bmin.y, v.y);
	//        bmin.z = min(bmin.z, v.z);
	//        bmax.x = max(bmax.x, v.x);
	//        bmax.y = max(bmax.y, v.y);
	//        bmax.z = max(bmax.z, v.z);
	//    }
	//}

	//glm::vec3 bsize = bmax - bmin;
	//float longestSize = max(bsize.x, max(bsize.y, bsize.z));

	//std::cout << "done scale : " << longestSize << std::endl;

	//std::ifstream pairReader("RT_pair_inversenormalRay.txt");

	//float value[6];
	//for (int i = 0; i < 6; i++) {
	//    pairReader >> value[i];
	//}
	//__pair[0] = { value[0], value[1], value[2] };
	//__pair[1] = { value[3], value[4], value[5] };

	//std::cout << __pair[0].x << " " << __pair[0].y << " " << __pair[0].z << std::endl;
	//std::cout << __pair[1].x << " " << __pair[1].y << " " << __pair[1].z << std::endl;

	//std::cout << "BOX" << std::endl;
	//std::cout << bmin.x << " " << bmin.y << " " << bmin.z << std::endl;
	//std::cout << bmax.x << " " << bmax.y << " " << bmax.z << std::endl;
	//std::cout << bsize.x << " " << bsize.y << " " << bsize.z << std::endl;
#pragma endregion

	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return 1;

	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	GLFWwindow* window = glfwCreateWindow(1280, 720, "OptiX Viewer", nullptr, nullptr);
	if (window == nullptr)
		return 1;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// imgui renderering
		ImGui::Begin("Hierarchy");
		{
			auto& launchData = optixRenderer.launchData;
			if (ImGui::TreeNode(std::string("Camera").c_str())) {
				Camera tmp = EditMode::getEditMode().camera;
				std::ostringstream oss;
				oss << "center : " << tmp.cen.x << " " << tmp.cen.y << " " << tmp.cen.z << std::endl;
				oss << "eye : " << tmp.eye.x << " " << tmp.eye.y << " " << tmp.eye.z << std::endl;
				oss << "up : " << tmp.up.x << " " << tmp.up.y << " " << tmp.up.z << std::endl;
				ImGui::Text(oss.str().c_str());
				ImGui::DragInt("Trace depth", &launchData.maxTrace, 1, 0, 31);
				ImGui::TreePop();
			}
		}
		{
			auto& light = optixRenderer.launchData.light;
			if (ImGui::TreeNode(std::string("Light").c_str())) {
				ImGui::DragFloat3("Position", &light.pos.x, 0.2f);
				ImGui::DragFloat3("color", &light.color.x, 0.2f, 0, 1);
				ImGui::DragFloat3("uv", &light.uv.x, 0.2f, 0, 1000.f);
				ImGui::DragFloat("Ambient", &light.ambient.x, 0.01f, 0.f, 1.f);

				ImGui::Checkbox("isSpot", &light.isSpot);
				if (light.isSpot) {
					ImGui::DragFloat3("spotDir", &light.spotDir.x, 0.2f);
					ImGui::DragFloat("angle", &light.angle.x, 0.2f, 0, 360.f);
				}
				ImGui::TreePop();
			}
		}
		{
			auto& tList = optixRenderer.transformationList;
			auto& matList = optixRenderer.materialList;

			bool isChangeTransform = false;
			bool isChangeMaterial = false;
			for (int i = 0; i < tList.size(); i++) {
				if (ImGui::TreeNode(std::string("Index : " + std::to_string(i)).c_str())) {
					isChangeMaterial = false;

					isChangeTransform |= ImGui::DragFloat3("Position", (float*)&tList[i].mTanslation, 0.2f);
					isChangeTransform |= ImGui::DragFloat3("Rotation", (float*)&tList[i].mRotation, 0.2f, 0, 360);
					isChangeTransform |= ImGui::DragFloat3("Scale", (float*)&tList[i].mScale, 0.2f, 0.1);

					isChangeMaterial |= ImGui::DragFloat3("color", &matList[i].color.x, 0.05, 0, 1);
					isChangeMaterial |= ImGui::DragFloat3("emission", &matList[i].emission.x, 0.05, 0, 1);
					isChangeMaterial |= ImGui::DragFloat("matallic", &matList[i].metallic, 0.01, 0, 1);
					isChangeMaterial |= ImGui::DragFloat("subsurface", &matList[i].subsurface, 0.01, 0, 1);
					isChangeMaterial |= ImGui::DragFloat("specular", &matList[i].specular, 0.01, 0, 1);
					isChangeMaterial |= ImGui::DragFloat("roughness", &matList[i].roughness, 0.01, 0, 1);
					isChangeMaterial |= ImGui::DragFloat("specularTint", &matList[i].specularTint, 0.01, 0, 1);
					isChangeMaterial |= ImGui::DragFloat("anisotropic", &matList[i].anisotropic, 0.01, 0, 1);
					isChangeMaterial |= ImGui::DragFloat("sheen", &matList[i].sheen, 0.01, 0, 1);
					isChangeMaterial |= ImGui::DragFloat("sheenTint", &matList[i].sheenTint, 0.01, 0, 1);
					isChangeMaterial |= ImGui::DragFloat("clearcoat", &matList[i].clearcoat, 0.01, 0, 1);
					isChangeMaterial |= ImGui::DragFloat("clearcoatGloss", &matList[i].clearcoatGloss, 0.01, 0, 1);
					isChangeMaterial |= ImGui::DragFloat("throughput", &matList[i].troughtput, 0.01, 0, 1);

					isChangeMaterial |= ImGui::Checkbox("isPlane", &matList[i].isPlane);
					if (matList[i].isPlane)
						isChangeMaterial |= ImGui::DragFloat("gridSize", &matList[i].gridSize, 0.05, 0.1f, 10.f);

					if (isChangeMaterial)
						optixRenderer.updateMaterial(i);

					ImGui::TreePop();
				}
			}
			if (isChangeTransform)
				optixRenderer.updateInstancesAS();
		}


		bool mouseEvent = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_ChildWindows)
			| ImGui::IsAnyItemFocused() | ImGui::IsAnyItemActive() | ImGui::IsItemHovered();
		if (!mouseEvent)
			EditMode::getEditMode().updateCamera();

		ImGui::End();
		ImGui::Render();

		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		optixRenderer.render(display_w, display_h);
		uint32_t* texture = new uint32_t[display_w * display_h];
		optixRenderer.downloadPixels(texture);

		if (textureID == -1)
			glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		GLenum texFormat = GL_RGBA;
		GLenum texelType = GL_UNSIGNED_BYTE;
		glTexImage2D(GL_TEXTURE_2D, 0, texFormat, display_w, display_h, 0, GL_RGBA, texelType, texture);

		glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		glDisable(GL_LIGHTING);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glDisable(GL_DEPTH_TEST);
		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.f, 0.f);
			glVertex3f(0.f, 0.f, 0.f);

			glTexCoord2f(0.f, 1.f);
			glVertex3f(0.f, (float)display_h, 0.f);

			glTexCoord2f(1.f, 1.f);
			glVertex3f((float)display_w, (float)display_h, 0.f);

			glTexCoord2f(1.f, 0.f);
			glVertex3f((float)display_w, 0.f, 0.f);
		}
		glEnd();

		glViewport(0, 0, display_w, display_h);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.f, (float)display_w, 0.f, (float)display_h, -1.f, 1.f);

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);
		delete[] texture;
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
