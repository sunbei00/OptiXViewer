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
//glBegin(GL_LINES);
//glColor3f(1, 1, 1);
//glVertex3f(__pair[0].x, __pair[0].y, __pair[0].z);
//glVertex3f(__pair[1].x, __pair[1].y, __pair[1].z);
//glEnd();

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int, char**) {
    auto objList = fastLoadOBJModels("../../penedepthSitu/bunny_case_03_oversampled.obj");
    OptiXRenderer optixRenderer;
    for (int i=0; i<objList.size(); i++)
        optixRenderer.createGeometryAS(*objList[i]);
    optixRenderer.createInstances();
    optixRenderer.createInstancesAS();
    optixRenderer.buildSBT();
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
        optixRenderer.render(display_w, display_h);

        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(c.eye.x, c.eye.y, c.eye.z, c.cen.x, c.cen.y, c.cen.z, c.up.x, c.up.y, c.up.z);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
