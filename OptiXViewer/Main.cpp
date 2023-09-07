#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "GL/glew.h"
#include "Camera.h"
#include <stdio.h>
#include <GLFW/glfw3.h>
#include "GL/glu.h"
#include <sstream>
#include "OptiXRenderer.h"
#include "Model.h"



static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int, char**) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "OptiX Viewer", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    OptiXRenderer optixRenderer;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);


    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // imgui renderering
        
        {
            Camera tmp = EditMode::getEditMode().camera;
            std::ostringstream oss;
            oss << "center : " << tmp.cen.x << " " << tmp.cen.y << " " << tmp.cen.z << std::endl;
            oss << "eye : " << tmp.eye.x << " " << tmp.eye.y << " " << tmp.eye.z << std::endl;
            oss << "up : " << tmp.up.x << " " << tmp.up.y << " " << tmp.up.z << std::endl;
            ImGui::Text(oss.str().c_str());
        }

        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        EditMode::getEditMode().updateCamera();
        Camera& c = EditMode::getEditMode().camera;

        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(c.eye.x, c.eye.y, c.eye.z, c.cen.x, c.cen.y, c.cen.z, c.up.x, c.up.y, c.up.z);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());\
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
