cmake_minimum_required(VERSION 3.20)

if(DEFINED SKBUILD_PROJECT_VERSION)
    set(SOLITONKIT_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
    set(SOLITONKIT_VERSION "0.1.0")
endif()

project(
    solitonkit
    VERSION ${SOLITONKIT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP QUIET)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else()
    message(STATUS "OpenMP not found, building without OpenMP")
endif()


# ============================================================
# Core C++ settings
# ============================================================

add_library(solitonkit_core INTERFACE)

target_include_directories(
    solitonkit_core
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(
        solitonkit_core
        INTERFACE
            OpenMP::OpenMP_CXX
    )

    target_compile_definitions(
        solitonkit_core
        INTERFACE
            SOLITONKIT_USE_OPENMP
    )
endif()

# ============================================================
# Tests
# ============================================================

option(SOLITONKIT_BUILD_TESTS "Build solitonkit tests" ON)

if(SOLITONKIT_BUILD_TESTS)
    enable_testing()

    file(GLOB_RECURSE SOLITONKIT_TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
    )

    foreach(TEST_SOURCE ${SOLITONKIT_TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

        add_executable(${TEST_NAME} ${TEST_SOURCE})

        target_link_libraries(
            ${TEST_NAME}
            PRIVATE
                solitonkit_core
        )

        add_test(
            NAME ${TEST_NAME}
            COMMAND ${TEST_NAME}
        )
    endforeach()
endif()


# ============================================================
# Python bindings with pybind11
# ============================================================

option(SOLITONKIT_BUILD_PYTHON "Build Python bindings" ON)

if(SOLITONKIT_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
    )

    if(NOT PYBIND11_CMAKE_DIR_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Could not find pybind11. Install it with:\n"
            "    python -m pip install pybind11"
        )
    endif()

    set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(
        _core
        python/bindings.cpp
    )

    target_link_libraries(
        _core
        PRIVATE
            solitonkit_core
    )

    target_include_directories(
        _core
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    if(SKBUILD)
        install(
            TARGETS _core
            LIBRARY DESTINATION solitonkit
            RUNTIME DESTINATION solitonkit
        )
    else()
        set(SOLITONKIT_PYTHON_PACKAGE_DIR
            ${CMAKE_CURRENT_SOURCE_DIR}/python/solitonkit
        )

        set_target_properties(
            _core
            PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY ${SOLITONKIT_PYTHON_PACKAGE_DIR}
                RUNTIME_OUTPUT_DIRECTORY ${SOLITONKIT_PYTHON_PACKAGE_DIR}
                ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()

    if(MSVC)
        target_compile_options(_core PRIVATE /W4)
    else()
        target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SOLITONKIT_BUILD_TESTS)
        add_test(
            NAME test_python_io_animation
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_io_animation.py"
                -v
        )

        set_tests_properties(
            test_python_io_animation
            PROPERTIES
                ENVIRONMENT
                    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python;MPLBACKEND=Agg"
        )
    endif()
endif()
