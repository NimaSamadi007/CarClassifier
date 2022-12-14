cmake_minimum_required(VERSION 3.10)

project(VECls)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("/home/nima/local-libs/onnxGPU/include")

link_directories("/home/nima/local-libs/onnxGPU/lib")

add_executable(${PROJECT_NAME} "src/main.cpp" "src/FScls.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} -lonnxruntime)
