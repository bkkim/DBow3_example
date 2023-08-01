#
cmake_minimum_required(VERSION 3.11)

#
project(
    DBoW3_exam
    VERSION 0.1
    DESCRIPTION "sample project"
    LANGUAGES CXX)

find_package(DBoW3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${DBoW3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(
    ${DBoW3_LIBRARY_DIRS}
    ${OpenCV_LIBRARY_DIRS}
)

file(GLOB source_files
    "src/*.cpp"
)

add_executable (${PROJECT_NAME}_main ${source_files})
target_compile_options(${PROJECT_NAME}_main PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_main ${OpenCV_LIBRARIES} ${DBoW3_LIBRARIES} ${OpenMP_CXX_FLAGS})