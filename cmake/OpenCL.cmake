option(ENABLE_OOO_OCL "Enable out-of-order (v2.0) OpenCL queue execution (default is ON, might not be available if OpenCL 1.x)" ON)

if (MSVC)
    set (OPENCL_AMDSDK "C:/Program Files (x86)/Common Files/Intel/Shared Libraries" CACHE PATH "Installation path for the installed AMD or Intel OpenCL SDK, if used")
else (MSVC)
    set (OPENCL_AMDSDK /opt/AMDAPPSDK-3.0 CACHE PATH "Installation path for the installed AMD or Intel OpenCL SDK, if used")
endif (MSVC)

# Options used when building the project
find_package (OpenCL)
if (NOT OpenCL_FOUND)
    # Attempt with AMD's OpenCL SDK
    if (MSVC)
        find_library (LIB_OPENCL OpenCL PATHS ${OPENCL_AMDSDK}/lib/)
    else (MSVC)
        find_library (LIB_OPENCL OpenCL PATHS ${OPENCL_AMDSDK}/lib/x86_64/)
    endif (MSVC)
    if (NOT LIB_OPENCL)
        set (ENABLE_OPENCL OFF)
        message ("Could not find OpenCL support from the AMD or Intel SDK (OPENCL_AMSDK directory not found)")
    elseif (NOT MSVC)
        # Found, set the required include path.
        set (OpenCL_INCLUDE_DIRS ${OPENCL_AMDSDK}/include CACHE PATH "AMD OpenCL SDK Header include path")
        set (OpenCL_COMPILATION_OPTIONS
            -Wno-ignored-attributes
            -Wno-deprecated-declarations
            CACHE STRING "AMD OpenCL SDK Compilation Option Requirements")
        message ("OpenCL support found in the AMD or Intel SDK (OPENCL_AMSDK directory variable)")
    endif()
endif ()

if (PACK_DEBIAN AND (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc"))
    set (ENABLE_OPENCL OFF)
endif (PACK_DEBIAN AND (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc"))

message ("OpenCL Support is: ${ENABLE_OPENCL}")

if (ENABLE_OPENCL)
    if (OpenCL_VERSION_MAJOR EQUAL 1)
        set(ENABLE_OOO_OCL OFF)
    endif (OpenCL_VERSION_MAJOR EQUAL 1)

    foreach (i IN ITEMS ${OpenCL_INCLUDE_DIRS})
        if (EXISTS ${i}/CL/opencl.hpp)
            set (OPENCL_V3 ON)
        endif ()
    endforeach ()
    if (OPENCL_V3)
        target_compile_definitions (qrack PUBLIC CL_HPP_TARGET_OPENCL_VERSION=300)
        target_compile_definitions (qrack PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=110)
    else (OPENCL_V3)
        if (ENABLE_OOO_OCL)
            target_compile_definitions (qrack PUBLIC CL_HPP_TARGET_OPENCL_VERSION=200)
        else (ENABLE_OOO_OCL)
            target_compile_definitions (qrack PUBLIC CL_HPP_TARGET_OPENCL_VERSION=120)
        endif (ENABLE_OOO_OCL)
        target_compile_definitions (qrack PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=110)
    endif (OPENCL_V3)

    if (ENABLE_SNUCL)
        find_package(MPI REQUIRED)
        set(QRACK_OpenCL_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH} $ENV{SNUCLROOT}/inc)
        set(QRACK_OpenCL_LINK_DIRS $ENV{SNUCLROOT}/lib)
        message(${QRACK_OpenCL_LINK_DIRS})
        set(QRACK_OpenCL_COMPILATION_OPTIONS ${MPI_CXX_COMPILE_FLAGS} ${OpenCL_COMPILATION_OPTIONS} -Wno-deprecated-declarations -Wno-ignored-attributes)
        find_library(
            SNUCL_CLUSTER
            NAMES snucl_cluster snucl_cpu snucl_single
            PATHS ${QRACK_OpenCL_LINK_DIRS}
            REQUIRED
        )
        target_link_libraries (qrack PUBLIC ${SNUCL_CLUSTER})
    else (ENABLE_SNUCL)
        set(QRACK_OpenCL_LIBRARIES ${OpenCL_LIBRARIES})
        set(QRACK_OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
        set(QRACK_OpenCL_LINK_DIRS ${OpenCL_LINK_DIRS})
        set(QRACK_OpenCL_COMPILATION_OPTIONS ${OpenCL_COMPILATION_OPTIONS})
    endif (ENABLE_SNUCL)

    message ("SnuCL Support is: ${ENABLE_SNUCL}")
    message ("    libOpenCL: ${QRACK_OpenCL_LIBRARIES}")
    message ("    Includes:  ${QRACK_OpenCL_INCLUDE_DIRS}")
    message ("    Options:   ${QRACK_OpenCL_COMPILATION_OPTIONS}")

    link_directories (${QRACK_OpenCL_LINK_DIRS})
    target_include_directories (qrack PUBLIC ${PROJECT_BINARY_DIR} ${QRACK_OpenCL_INCLUDE_DIRS})
    target_compile_options (qrack PUBLIC ${QRACK_OpenCL_COMPILATION_OPTIONS})
    target_link_libraries (qrack PUBLIC ${QRACK_OpenCL_LIBRARIES})

    # Build the OpenCL command files
    find_program (XXD_BIN xxd)
    file (GLOB_RECURSE COMPILABLE_RESOURCES "src/common/*.cl")
    foreach (INPUT_FILE ${COMPILABLE_RESOURCES})
        get_filename_component (INPUT_NAME ${INPUT_FILE} NAME)
        get_filename_component (INPUT_BASENAME ${INPUT_FILE} NAME_WE)
        get_filename_component (INPUT_DIR ${INPUT_FILE} DIRECTORY)

        set (OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/include/common/${INPUT_BASENAME}cl.hpp)

        message (" Creating XXD Rule for ${INPUT_FILE} -> ${OUTPUT_FILE}")
        add_custom_command (
            WORKING_DIRECTORY ${INPUT_DIR}
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${XXD_BIN} -i ${INPUT_NAME} > ${OUTPUT_FILE}
            COMMENT "Building OpenCL Commands in ${INPUT_FILE}"
            )
        list (APPEND COMPILED_RESOURCES ${OUTPUT_FILE})
    endforeach ()

    # Add the OpenCL objects to the library
    target_sources (qrack PRIVATE
        ${COMPILED_RESOURCES}
        src/common/oclengine.cpp
        src/qengine/opencl.cpp
        src/qhybrid.cpp
        src/qunitmulti.cpp
        )

    if (APPLE OR CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc")
        include(FetchContent)
        FetchContent_Declare (OpenCL-Headers
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers
            GIT_TAG v2024.05.08
        )
        FetchContent_Declare (OpenCL-ICD-Loader
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader
            GIT_TAG v2024.05.08
        )
        FetchContent_Declare (OpenCL-CLHPP
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP
            GIT_TAG v2024.05.08
        )
        FetchContent_MakeAvailable(OpenCL-Headers OpenCL-ICD-Loader OpenCL-CLHPP)
        target_include_directories (qrack PUBLIC ${CMAKE_BIN_DIR}/_deps/opencl-headers-src/ ${CMAKE_BIN_DIR}/_deps/opencl-clhpp-src/include/)
    endif (APPLE OR CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc")

endif (ENABLE_OPENCL)
