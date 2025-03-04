cmake_minimum_required(VERSION 3.11)
project(franka_control VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -flto=auto")

# Add these lines for RPATH handling
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)
find_package(Franka REQUIRED)

if (${Franka_VERSION} VERSION_GREATER_EQUAL "0.8.0")
    add_compile_definitions(FRANKA_0_8)
endif ()

if (${Franka_VERSION} VERSION_GREATER_EQUAL "0.9.0")
    add_compile_definitions(FRANKA_0_9)
endif ()

message("Found Eigen Version: ${Eigen3_VERSION}")
message("Found Franka Version: ${Franka_VERSION}")

file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/franka_control.cpp")
add_library(franka_control SHARED ${SRC})
target_include_directories(franka_control PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(franka_control PUBLIC Franka::Franka Eigen3::Eigen)

set(PYBIND11_FINDPYTHON ON)
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_ROOT_DIR $ENV{CONDA_PREFIX})

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Build Python module
file(GLOB_RECURSE PYTHON_SRC "${CMAKE_CURRENT_SOURCE_DIR}/python.cpp")
pybind11_add_module(_franka-control-lib ${PYTHON_SRC})
target_link_libraries(_franka-control-lib PRIVATE franka_control)
set_target_properties(_franka-control-lib PROPERTIES OUTPUT_NAME "_franka_control")

# Set include directories explicitly
target_include_directories(_franka-control-lib PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set the output directory for both libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Install both libraries to the same location
install(TARGETS _franka-control-lib franka_control
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

