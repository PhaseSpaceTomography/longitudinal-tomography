cmake_minimum_required(VERSION 3.16)
project(tomo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

include(FindOpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

exec_program(${Python_EXECUTABLE}
        ARGS "-c \"import pybind11; print(pybind11.get_include())\""
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        RETURN_VALUE PYBIND11_NOT_FOUND
        )

if (PYBIND11_NOT_FOUND)
    message(FATAL_ERROR "PyBind11 headers not found")
endif ()

include_directories(${Python_INCLUDE_DIRS})
include_directories(${PYBIND11_INCLUDE_DIR})

add_library(tomo SHARED
        tomo/cpp_routines/libtomo.cpp
        tomo/cpp_routines/kick_and_drift.cpp
        tomo/cpp_routines/reconstruct.cpp
        tomo/cpp_routines/data_treatment.cpp)

target_link_libraries(tomo ${Python_LIBRARIES})

set_target_properties(tomo
        PROPERTIES
        PREFIX ""
        OUTPUT_NAME "libtomo")

add_executable(placeholder
        tomo/cpp_routines/libtomo.cpp
        tomo/cpp_routines/kick_and_drift.cpp
        tomo/cpp_routines/reconstruct.cpp
        tomo/cpp_routines/data_treatment.cpp)