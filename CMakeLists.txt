cmake_minimum_required (VERSION 2.6)
project (fftwater)

find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Boost 1.67.0 COMPONENTS thread system)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
add_executable (fftwater "${PROJECT_SOURCE_DIR}/src/main.cpp")
set_property(TARGET fftwater PROPERTY CXX_STANDARD 17)
set_source_files_properties( "${PROJECT_SOURCE_DIR}/src/main.cpp" PROPERTIES COMPILE_FLAGS "-g -DGLEW_STATIC -O3" )

set(GLFW_DIR "${PROJECT_SOURCE_DIR}/glfw")
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW example programs")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW test programs")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
set(GLFW_INSTALL OFF CACHE INTERNAL "Generate installation target")
add_subdirectory("${GLFW_DIR}")
target_link_libraries(fftwater "glfw" "${GLFW_STATIC_LIBRARIES}")
target_include_directories(fftwater PRIVATE "${GLFW_DIR}/include")
target_compile_definitions(fftwater PRIVATE "GLFW_INCLUDE_NONE")

target_include_directories(fftwater PRIVATE "${GLEW_INCLUDE_DIR}" "${OPENGL_INCLUDE_DIRS}")
target_link_libraries(fftwater "${GLEW_LIBRARIES}" "${OPENGL_LIBRARIES}")

set(GLM_DIR "${PROJECT_SOURCE_DIR}/glm")
target_include_directories(fftwater PRIVATE "${GLM_DIR}")
target_include_directories(fftwater PRIVATE "${Boost_INCLUDE_DIR}")

link_directories(${Boost_LIBRARY_DIR})
target_link_libraries(fftwater ${Boost_LIBRARIES})
