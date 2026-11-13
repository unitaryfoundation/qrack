option (ENABLE_CUDA "Build CUDA-based QEngine type" OFF)
option (ENABLE_GVIRTUS "Link against GVirtuS for distributed CUDA" OFF)
# Compile the CUDA QEngine backend as HIP for AMD ROCm GPUs instead of nvcc.
# The single backend source is shared; only the toolchain differs.
option (USE_HIP "Build the CUDA QEngine backend for AMD ROCm via HIP" OFF)

if (ENABLE_OPENCL)
    set(ENABLE_CUDA OFF)
endif ()

if (ENABLE_CUDA AND NOT USE_HIP)
    if ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        find_package(CUDA)
        if (NOT CUDA_FOUND)
            set(ENABLE_CUDA OFF)
        endif ()
    else ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        find_package (CUDAToolkit)
        if (NOT CUDAToolkit_FOUND)
            set(ENABLE_CUDA OFF)
        endif ()
    endif ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
endif()

message ("CUDA Support is: ${ENABLE_CUDA}")
if (ENABLE_CUDA AND USE_HIP)
    message ("HIP (ROCm) backend is: ON")
else ()
    message ("HIP (ROCm) backend is: OFF")
endif ()

if (ENABLE_CUDA AND USE_HIP)
    # enable_language(HIP) honors an explicit -DCMAKE_HIP_ARCHITECTURES, otherwise
    # auto-detects the host GPU(s), and errors on a no-GPU build host.
    enable_language(HIP)
    target_compile_definitions(qrack PUBLIC ENABLE_CUDA=1 USE_HIP=1)
    message("CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")

    find_package(hip REQUIRED)
    # qrack_types.hpp pulls the HIP vector types (float2/float4/make_float2) into
    # every TU, including the host C++ ones the default compiler builds; give the
    # host compiler the ROCm headers and AMD platform define so the shim resolves.
    target_link_libraries (qrack PUBLIC hip::host)
    target_compile_definitions (qrack PUBLIC __HIP_PLATFORM_AMD__=1)
    target_include_directories (qrack PUBLIC ${PROJECT_BINARY_DIR})
    # Targets created before enable_language(HIP) did not see auto-detected
    # architectures, and every target linking qrack links as HIP.
    list(REMOVE_DUPLICATES CMAKE_HIP_ARCHITECTURES)
    get_property(QRACK_HIP_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    set_target_properties(${QRACK_HIP_TARGETS} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

    # The CUDA backend sources compile unchanged as HIP via the cuda_to_hip.h shim.
    set_source_files_properties(
        src/common/cudaengine.cu
        src/common/qengine.cu
        src/qengine/cuda.cu
        PROPERTIES LANGUAGE HIP)

    # Add the CUDA (HIP) objects to the library
    target_sources (qrack PRIVATE
        src/common/cudaengine.cu
        src/common/qengine.cu
        src/qengine/cuda.cu
        src/qhybrid.cpp
        src/qunitmulti.cpp
        )

elseif (ENABLE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(qrack PUBLIC ENABLE_CUDA=1)
    if ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        target_include_directories (qrack PUBLIC ${PROJECT_BINARY_DIR} ${CUDA_INCLUDE_DIRS})
        target_compile_options (qrack PUBLIC ${CUDA_COMPILATION_OPTIONS})
        set(QRACK_CUDA_LIBRARIES ${CUDA_LIBRARIES})
    else ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        target_include_directories (qrack PUBLIC ${PROJECT_BINARY_DIR} ${CUDAToolkit_INCLUDE_DIRS})
        target_compile_options (qrack PUBLIC ${CUDAToolkit_COMPILATION_OPTIONS})
        set(QRACK_CUDA_LIBRARIES ${CUDAToolkit_LIBRARIES})
    endif ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))

    if (NOT DEFINED QRACK_CUDA_ARCHITECTURES)
        # See https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures#answer-68223399
        if (${CMAKE_VERSION} VERSION_LESS "3.24.0")
            include(FindCUDA/select_compute_arch)
            CUDA_DETECT_INSTALLED_GPUS(QRACK_CUDA_ARCHITECTURES)
            string(STRIP "${QRACK_CUDA_ARCHITECTURES}" QRACK_CUDA_ARCHITECTURES)
            string(REPLACE " " ";" QRACK_CUDA_ARCHITECTURES "${QRACK_CUDA_ARCHITECTURES}")
            string(REPLACE "." "" QRACK_CUDA_ARCHITECTURES "${QRACK_CUDA_ARCHITECTURES}")
        else (${CMAKE_VERSION} VERSION_LESS "3.24.0")
            set(QRACK_CUDA_ARCHITECTURES native)
        endif (${CMAKE_VERSION} VERSION_LESS "3.24.0")
    endif (NOT DEFINED QRACK_CUDA_ARCHITECTURES)

    message("QRACK_CUDA_ARCHITECTURES: ${QRACK_CUDA_ARCHITECTURES}")

    target_link_libraries (qrack ${QRACK_CUDA_LIBRARIES})
    set_target_properties(qrack PROPERTIES CUDA_ARCHITECTURES "${QRACK_CUDA_ARCHITECTURES}")
    
    # Add the CUDA objects to the library
    target_sources (qrack PRIVATE
        src/common/cudaengine.cu
        src/common/qengine.cu
        src/qengine/cuda.cu
        src/qhybrid.cpp
        src/qunitmulti.cpp
        )

endif ()
