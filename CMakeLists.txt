cmake_minimum_required (VERSION 3.10)
project (Qrack VERSION 9.32.6 DESCRIPTION "High Performance Quantum Bit Simulation" LANGUAGES CXX)

# Installation commands
include (GNUInstallDirs)

include ("cmake/Coverage.cmake")
include ("cmake/Format.cmake")
include ("cmake/EmitLlvm.cmake")
# PSTRIDEPOW relates to parallel for loops:
include ("cmake/Pthread.cmake")

include_directories ("include" "include/common" "${CMAKE_BINARY_DIR}/include/common")

set(CMAKE_WARN_DEPRECATED FALSE)

option (ENABLE_MAKE_CLEAN_OVERRIDE "Override the 'make clean' recipe for Debian packaging" ON)
option (ENABLE_EXAMPLES "Include usage examples" ON)
option (ENABLE_INTRINSICS "Use x86_64 intrinsics" ON)
if (PACK_DEBIAN)
    set(ENABLE_EXAMPLES OFF)
endif (PACK_DEBIAN)

# Declare the library
add_library (qrack STATIC
    src/common/functions.cpp
    src/common/parallel_for.cpp
    src/qinterface/arithmetic.cpp
    src/qinterface/gates.cpp
    src/qinterface/logic.cpp
    src/qinterface/qinterface.cpp
    src/qinterface/rotational.cpp
    src/qengine/qengine.cpp
    src/qengine/arithmetic.cpp
    src/qengine/state.cpp
    src/qengine/utility.cpp
    src/qengineshard.cpp
    src/qunit.cpp
    src/qunitclifford.cpp
    src/qpager.cpp
    src/qstabilizer.cpp
    src/qstabilizerhybrid.cpp
    src/qcircuit.cpp
    src/qtensornetwork.cpp
    src/qinterface_noisy.cpp
    )
set_property(TARGET qrack PROPERTY POSITION_INDEPENDENT_CODE ON)
if (ENABLE_PTHREAD AND ENABLE_QUNIT_CPU_PARALLEL)
    target_sources (qrack PRIVATE
        src/common/dispatchqueue.cpp
        )
endif (ENABLE_PTHREAD AND ENABLE_QUNIT_CPU_PARALLEL)

if (ENABLE_EMIT_LLVM)
    add_library (qrack_pinvoke OBJECT
        src/pinvoke_api.cpp
        )
else (ENABLE_EMIT_LLVM)
    add_library (qrack_pinvoke SHARED
        src/pinvoke_api.cpp
        )
endif (ENABLE_EMIT_LLVM)
set_property(TARGET qrack_pinvoke PROPERTY POSITION_INDEPENDENT_CODE ON)

if (EMSCRIPTEN)
    add_library (qrack_wasm STATIC
        src/wasm_api.cpp
        )
    set_property(TARGET qrack_wasm PROPERTY POSITION_INDEPENDENT_CODE ON)
endif (EMSCRIPTEN)

set(QRACK_LIBS qrack)
if (ENABLE_PTHREAD)
    if (ANDROID OR CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv" OR (NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^arm"))
        set(QRACK_LIBS ${QRACK_LIBS} atomic)
    endif (ANDROID OR CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv" OR (NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^arm"))
    if (NOT ANDROID AND NOT MSVC)
	set(QRACK_LIBS ${QRACK_LIBS} pthread)
    endif (NOT ANDROID AND NOT MSVC)
endif (ENABLE_PTHREAD)

target_link_libraries (qrack_pinvoke ${QRACK_LIBS})
if (EMSCRIPTEN)
    target_link_libraries (qrack_wasm ${QRACK_LIBS})
endif (EMSCRIPTEN)
if (NOT ENABLE_EMIT_LLVM AND NOT PACK_DEBIAN)
    # Declare the unittest executable
    add_executable (unittest
        test/test_main.cpp
        test/tests.cpp
        )

    target_link_libraries (unittest ${QRACK_LIBS})

    # Declare the benchmark executable
    add_executable (benchmarks
        test/benchmarks_main.cpp
        test/benchmarks.cpp
        )

    target_link_libraries (benchmarks ${QRACK_LIBS})

endif (NOT ENABLE_EMIT_LLVM AND NOT PACK_DEBIAN)

# Included after the library and other modules have been declared
option (ENABLE_OPENCL "Use OpenCL optimizations" ON)
option (ENABLE_SNUCL "Use SnuCL framework for clusters" OFF)
include ("cmake/Alu.cmake")
include ("cmake/Bcd.cmake")
include ("cmake/Boost.cmake")
include ("cmake/Complex_x2.cmake")
include ("cmake/CppStd.cmake")
include ("cmake/EnvVars.cmake")
include ("cmake/FpMath.cmake")
include ("cmake/OclMemGuards.cmake")
include ("cmake/Pstridepow.cmake")
include ("cmake/Pure32.cmake")
include ("cmake/UIntPow.cmake")
include ("cmake/QbCapPow.cmake")
include ("cmake/Qbdt.cmake")
include ("cmake/RegGates.cmake")
include ("cmake/RotApi.cmake")
include ("cmake/VM6502Q.cmake")
if (ENABLE_EXAMPLES AND NOT ENABLE_EMIT_LLVM)
    include ("cmake/Examples.cmake")
endif (ENABLE_EXAMPLES AND NOT ENABLE_EMIT_LLVM)
include ("cmake/OpenCL.cmake" )
include ("cmake/CUDA.cmake")

message ("C++ standard year is (at least): ${CPP_STD}")
message ("Qubit capacity power is: ${QBCAPPOW}")
message ("Floating-point capacity power is: ${FPPOW}")
message ("UInt capacity power is: ${UINTPOW}")
message ("CPU parallelism is: ${ENABLE_PTHREAD}")
message ("CPU work item stride power is: ${PSTRIDEPOW}")
message ("Parallel QUnit->CPU is: ${ENABLE_QUNIT_CPU_PARALLEL}")
message ("Parallel QBdt is: ${ENABLE_QBDT_CPU_PARALLEL}")
message ("Pure 32-bit compilation is: ${ENABLE_PURE32}")
message ("Complex_x2/AVX support is: ${ENABLE_COMPLEX_X2}")
message ("SSE3.0 support is: ${ENABLE_SSE3}")
message ("OpenCL memory guards are: ${ENABLE_OCL_MEM_GUARDS}")
message ("OpenCL (v2.0) out-of-order queue is: ${ENABLE_OOO_OCL}")
message ("Quantum Binary decision tree (QBDT) inclusion is: ${ENABLE_QBDT}")
message ("General ALU API inclusion is: ${ENABLE_ALU}")
message ("BCD ALU API inclusion is: ${ENABLE_BCD}")
message ("Extended rotation API inclusion is: ${ENABLE_ROT_API}")
message ("Register-spanning API inclusion is: ${ENABLE_REG_GATES}")
message ("Environment variable usage is: ${ENABLE_ENV_VARS}")
message ("VM6502Q disassembler support is: ${ENABLE_VM6502Q_DEBUG}")
message ("Compiling libraries to LLVM IR is: ${ENABLE_EMIT_LLVM}")

if (FPPOW GREATER 6)
    target_link_libraries(qrack PUBLIC quadmath)
endif (FPPOW GREATER 6)

if (ENABLE_COMPLEX_X2)
    set(COMPLEX_X2_MACRO "(!(defined(__GNUC__) || defined(__MINGW32__)) || ((FPPOW == 5) && __SSE__) || ((FPPOW == 6) && __SSE2__))")
else (ENABLE_COMPLEX_X2)
    set(COMPLEX_X2_MACRO "0")
endif (ENABLE_COMPLEX_X2)

if (ENABLE_SSE3)
    set(SSE3_MACRO "(!(defined(__GNUC__) || defined(__MINGW32__)) || __SSE3__)")
else (ENABLE_SSE3)
    set(SSE3_MACRO "0")
endif (ENABLE_SSE3)

if (MSVC)
    if (CPP_STD GREATER_EQUAL 23)
        set(QRACK_CPP_STD_OPT /std:c++23)
    elseif (CPP_STD GREATER_EQUAL 20)
        set(QRACK_CPP_STD_OPT /std:c++20)
    elseif (CPP_STD GREATER_EQUAL 17)
        set(QRACK_CPP_STD_OPT /std:c++17)
    elseif (CPP_STD GREATER_EQUAL 13)
        set(QRACK_CPP_STD_OPT /std:c++14)
    else ()
        set(QRACK_CPP_STD_OPT "")
    endif ()
else (MSVC)
    if (CPP_STD GREATER_EQUAL 23)
        set(QRACK_CPP_STD_OPT -std=c++23)
    elseif (CPP_STD GREATER_EQUAL 20)
        set(QRACK_CPP_STD_OPT -std=c++20)
    elseif (CPP_STD GREATER_EQUAL 17)
        set(QRACK_CPP_STD_OPT -std=c++17)
    elseif (CPP_STD GREATER_EQUAL 13)
        set(QRACK_CPP_STD_OPT -std=c++14)
    else ()
        set(QRACK_CPP_STD_OPT -std=c++11)
    endif ()
endif (MSVC)

if (QBCAPPOW GREATER 6)
    set(QRACK_CUDA_COMPILE_OPTS -O3 -use_fast_math -Xcompiler -fpermissive ${QRACK_CPP_STD_OPT} --ptxas-options -O3,)
else (QBCAPPOW GREATER 6)
    set(QRACK_CUDA_COMPILE_OPTS -O3 -use_fast_math -Werror all-warnings ${QRACK_CPP_STD_OPT} --ptxas-options -O3,)
endif (QBCAPPOW GREATER 6)

if (ENABLE_GVIRTUS)
    set(QRACK_CUDA_COMPILE_OPTS --cudart=shared ${QRACK_CUDA_COMPILE_OPTS})
endif (ENABLE_GVIRTUS)

if (MSVC)
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
 
        set(QRACK_COMPILE_FLAGS_PART /O2 ${QRACK_CPP_STD_OPT} /Wall)
    else (CMAKE_BUILD_TYPE STREQUAL "Release")
        set(QRACK_COMPILE_FLAGS_PART ${QRACK_CPP_STD_OPT} /Wall)
    endif (CMAKE_BUILD_TYPE STREQUAL "Release")
else (MSVC)
    set(QRACK_COMPILE_FLAGS_PART -O3 ${QRACK_CPP_STD_OPT})
    if (EMSCRIPTEN AND ENABLE_PTHREAD)
        set (QRACK_COMPILE_FLAGS_PART ${QRACK_COMPILE_FLAGS_PART} -pthread)
    endif (EMSCRIPTEN AND ENABLE_PTHREAD)
    set(QRACK_COMPILE_FLAGS_PART ${QRACK_COMPILE_FLAGS_PART} -Wall)
    if (EMSCRIPTEN OR NOT APPLE OR NOT ENABLE_OPENCL)
        set(QRACK_COMPILE_FLAGS_PART ${QRACK_COMPILE_FLAGS_PART} -Werror)
    endif (EMSCRIPTEN OR NOT APPLE OR NOT ENABLE_OPENCL)
endif(MSVC)

set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_FLAGS_PART})
set(TEST_COMPILE_OPTS ${QRACK_COMPILE_FLAGS_PART})

include(CheckCXXCompilerFlag)
if (ENABLE_COMPLEX_X2 AND FPPOW GREATER 5)
    check_cxx_compiler_flag(-mavx COMPILER_SUPPORTS_AVX_FLAG)
    if (COMPILER_SUPPORTS_AVX_FLAG)
        set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -mavx)
    endif (COMPILER_SUPPORTS_AVX_FLAG)
endif (ENABLE_COMPLEX_X2 AND FPPOW GREATER 5)

if (NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^amd"))
    set(ENABLE_INTRINSICS OFF)
    set(ENABLE_COMPLEX_X2 OFF)
    set(ENABLE_SSE3 OFF)
endif (NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^amd"))

if (ENABLE_SSE3)
    check_cxx_compiler_flag(-msse3 COMPILER_SUPPORTS_SSE3_FLAG)
    if (COMPILER_SUPPORTS_SSE3_FLAG AND (NOT APPLE))
        set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -msse3)
    endif (COMPILER_SUPPORTS_SSE3_FLAG AND (NOT APPLE))
endif (ENABLE_SSE3)

if (FPPOW LESS 5)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^arm" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -mfp16-format=ieee)
        set(TEST_COMPILE_OPTS ${TEST_COMPILE_OPTS} -mfp16-format=ieee)
    endif ()
endif (FPPOW LESS 5)

if (EMSCRIPTEN AND ENABLE_COMPLEX_X2)
    check_cxx_compiler_flag(-msimd128 COMPILER_SUPPORTS_FLAG)
    if (COMPILER_SUPPORTS_FLAG)
        set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -msimd128)
    endif (COMPILER_SUPPORTS_FLAG)
    check_cxx_compiler_flag(-msse COMPILER_SUPPORTS_FLAG)
    if (COMPILER_SUPPORTS_FLAG)
        set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -msse)
    endif (COMPILER_SUPPORTS_FLAG)
    check_cxx_compiler_flag(-msse2 COMPILER_SUPPORTS_FLAG)
    if (COMPILER_SUPPORTS_FLAG)
        set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -msse2)
    endif (COMPILER_SUPPORTS_FLAG)
endif (EMSCRIPTEN AND ENABLE_COMPLEX_X2)

if (ENABLE_EMIT_LLVM)
    set(QRACK_COMPILE_OPTS ${QRACK_COMPILE_OPTS} -emit-llvm)
endif (ENABLE_EMIT_LLVM)

include ("cmake/Random.cmake")

include_directories(include/common)

if (NOT ENABLE_EMIT_LLVM AND NOT PACK_DEBIAN)
    target_include_directories (unittest PUBLIC test)
    target_include_directories (benchmarks PUBLIC test)
endif (NOT ENABLE_EMIT_LLVM AND NOT PACK_DEBIAN)

if (APPLE)
    set(TEST_COMPILE_OPTS -Wno-inconsistent-missing-override)
endif (APPLE)

if (ENABLE_CUDA)
    target_compile_options (qrack PUBLIC
        "$<$<COMPILE_LANGUAGE:C>:${QRACK_COMPILE_OPTS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${QRACK_COMPILE_OPTS}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${QRACK_CUDA_COMPILE_OPTS}>"
        -DCATCH_CONFIG_FAST_COMPILE
        )
    target_compile_options (qrack_pinvoke PUBLIC
        "$<$<COMPILE_LANGUAGE:C>:${QRACK_COMPILE_OPTS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${QRACK_COMPILE_OPTS}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${QRACK_CUDA_COMPILE_OPTS}>"
        -DCATCH_CONFIG_FAST_COMPILE
        )
    if (EMSCRIPTEN)
        target_compile_options (qrack_wasm PUBLIC
            "$<$<COMPILE_LANGUAGE:C>:${QRACK_COMPILE_OPTS}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${QRACK_COMPILE_OPTS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:${QRACK_CUDA_COMPILE_OPTS}>"
            -DCATCH_CONFIG_FAST_COMPILE
            )
    endif (EMSCRIPTEN)
else (ENABLE_CUDA)
    if (MSVC)
        target_compile_options (qrack PUBLIC ${QRACK_COMPILE_OPTS})
    else (MSVC)
        target_compile_options (qrack PUBLIC ${QRACK_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
    endif (MSVC)
    target_compile_options (qrack_pinvoke PUBLIC ${QRACK_COMPILE_OPTS})
    if (EMSCRIPTEN)
        target_compile_options (qrack_wasm PUBLIC ${QRACK_COMPILE_OPTS})
    endif (EMSCRIPTEN)
endif (ENABLE_CUDA)
target_compile_definitions(qrack_pinvoke PUBLIC -DDLL_EXPORTS)
set_property(TARGET qrack_pinvoke PROPERTY SOVERSION ${CMAKE_PROJECT_VERSION} )
if (NOT ENABLE_EMIT_LLVM AND NOT PACK_DEBIAN)
    target_compile_options (unittest PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
    target_compile_options (benchmarks PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
endif (NOT ENABLE_EMIT_LLVM AND NOT PACK_DEBIAN)

if (NOT ENABLE_EMIT_LLVM)
    # Declare the OCL precompilation executable
    add_executable (qrack_cl_precompile
        src/qrack_cl_precompile.cpp
        )
    target_link_libraries (qrack_cl_precompile ${QRACK_LIBS})
    target_compile_options (qrack_cl_precompile PUBLIC ${TEST_COMPILE_OPTS})
    install(TARGETS qrack_cl_precompile RUNTIME DESTINATION bin BUNDLE DESTINATION bin PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif (NOT ENABLE_EMIT_LLVM)

set_target_properties (qrack PROPERTIES
    VERSION ${PROJECT_VERSION}
    )

set_target_properties (qrack_pinvoke PROPERTIES
    VERSION ${PROJECT_VERSION}
    )

if (EMSCRIPTEN)
    set_target_properties (qrack_wasm PROPERTIES
        VERSION ${PROJECT_VERSION}
        )
endif (EMSCRIPTEN)

# Install common headers
install (FILES
    ${CMAKE_BINARY_DIR}/include/common/config.h
    include/common/qrack_functions.hpp
    include/common/qrack_types.hpp
    include/common/complex16x2simd.hpp
    include/common/complex8x2simd.hpp
    include/common/oclapi.hpp
    include/common/oclengine.hpp
    include/common/cudaengine.cuh
    include/common/cuda_kernels.cuh
    include/common/parallel_for.hpp
    include/common/rdrandwrapper.hpp
    include/common/cuda_kernels.cuh
    include/common/dispatchqueue.hpp
    include/common/big_integer.hpp
    include/common/half.hpp
    include/common/qneuron_activation_function.hpp
    include/common/pauli.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qrack/common
    )

# Install qrack library headers
install (FILES
    include/hamiltonian.hpp
    include/statevector.hpp
    include/pinvoke_api.hpp
    include/wasm_api.hpp
    include/qcircuit.hpp
    include/qfactory.hpp
    include/qengine.hpp
    include/qengine_cpu.hpp
    include/qubitswapmap.hpp
    include/qengineshard.hpp
    include/mpsshard.hpp
    include/qunit.hpp
    include/qunitclifford.hpp
    include/qunitmulti.hpp
    include/qunitstatevector.hpp
    include/qengine_gpu_util.hpp
    include/qengine_opencl.hpp
    include/qengine_cuda.hpp
    include/qtensornetwork.hpp
    include/qinterface.hpp
    include/qinterface_noisy.hpp
    include/qalu.hpp
    include/qparity.hpp
    include/qneuron.hpp
    include/qpager.hpp
    include/qhybrid.hpp
    include/qstabilizer.hpp
    include/qstabilizerhybrid.hpp
    include/qbdt.hpp
    include/qbdt_node.hpp
    include/qbdt_node_interface.hpp
    include/qbdthybrid.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/qrack
    )

# Install the archive
install (TARGETS qrack
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/qrack
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

if (EMSCRIPTEN)
    install (TARGETS qrack_wasm
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/qrack
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
endif (EMSCRIPTEN)

if (NOT ENABLE_EMIT_LLVM)
    install (TARGETS qrack_pinvoke
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/qrack
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE
        )
endif (NOT ENABLE_EMIT_LLVM)

# Install the pkg-config file
configure_file (libqrack.pc.in libqrack.pc @ONLY)
install (FILES ${CMAKE_BINARY_DIR}/libqrack.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
install (FILES ${CMAKE_SOURCE_DIR}/debian/copyright DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/qrack)
if (NOT MSVC)
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/changelog.gz"
        COMMAND gzip -cn9 "${CMAKE_SOURCE_DIR}/debian/changelog" > "${CMAKE_BINARY_DIR}/changelog.gz"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        DEPENDS "${CMAKE_SOURCE_DIR}/debian/changelog"
        COMMENT "Compressing changelog"
    )
    add_custom_target(changelog ALL DEPENDS "${CMAKE_BINARY_DIR}/changelog.gz")
    install(FILES "${CMAKE_BINARY_DIR}/changelog.gz"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/qrack"
    )
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/qrack_cl_precompile.1.gz"
        COMMAND gzip -cn9 "${CMAKE_SOURCE_DIR}/debian/docs/qrack_cl_precompile.1" > "${CMAKE_BINARY_DIR}/qrack_cl_precompile.1.gz"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        DEPENDS "${CMAKE_SOURCE_DIR}/debian/docs/qrack_cl_precompile.1"
        COMMENT "Compressing qrack_cl_precompile.1"
    )
    add_custom_target(qrack_cl_precompile.1 ALL DEPENDS "${CMAKE_BINARY_DIR}/qrack_cl_precompile.1.gz")
    install(FILES "${CMAKE_BINARY_DIR}/qrack_cl_precompile.1.gz"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man1"
    )
endif (NOT MSVC)

include(InstallRequiredSystemLibraries)
set(CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    )
set(CPACK_PACKAGE_NAME "libqrack")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.md")
file(COPY LICENSE.md DESTINATION ${CMAKE_BINARY_DIR})
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
file(COPY README.md DESTINATION ${CMAKE_BINARY_DIR})
set(CPACK_PACKAGE_VERSION_MAJOR "${Qrack_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Qrack_VERSION_MINOR}")
set(CPACK_PACKAGE_VENDOR "vm6502q")
set(CPACK_PACKAGE_CONTACT "dan@unitary.fund")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "unitaryfund/qrack quantum computer simulator library")
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "unitaryfund/qrack (or vm6502q/qrack) quantum computer simulator library")
set(CPACK_STRIP_FILES TRUE)
set(CPACK_THREADS 0)

if (PACK_DEBIAN)
    file(COPY cmake DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY CMakeLists.txt DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY libqrack.pc.in DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY include DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY src DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY test DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY examples DESTINATION ${CMAKE_BINARY_DIR})

    set(CPACK_SOURCE_GENERATOR "DEB")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Daniel Strano <dan@unitary.fund>")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/unitaryfund/qrack")

    file(READ debian/changelog DEBIAN_CHANGELOG)
    string(REPLACE ") " ";" VERSION_LIST ${DEBIAN_CHANGELOG})
    list(GET VERSION_LIST 1 DEBIAN_CHANGELOG)
    string(REPLACE " " ";" VERSION_LIST ${DEBIAN_CHANGELOG})
    list(GET VERSION_LIST 0 DEBIAN_DISTRO_SERIES)
    message("Debian distroseries: ${DEBIAN_DISTRO_SERIES}")

    if (DEBHELPER_COMPAT_VERSION)
        message("debhelper-compat version override: ${DEBHELPER_COMPAT_VERSION}")
    else (DEBHELPER_COMPAT_VERSION)
        set(DEBHELPER_COMPAT_VERSION "13")
        if (DEBIAN_DISTRO_SERIES STREQUAL "bionic")
            set(DEBHELPER_COMPAT_VERSION "11")
        endif (DEBIAN_DISTRO_SERIES STREQUAL "bionic")
        if (DEBIAN_DISTRO_SERIES STREQUAL "focal")
            set(DEBHELPER_COMPAT_VERSION "12")
        endif (DEBIAN_DISTRO_SERIES STREQUAL "focal")
    endif (DEBHELPER_COMPAT_VERSION)
    if (DEBHELPER_COMPAT_VERSION STREQUAL "13")
        set(OPENCL_V3 ON)
        set(CMAKE_INSTALL_LIBDIR "/usr/lib")
    else (DEBHELPER_COMPAT_VERSION STREQUAL "13")
        set(OPENCL_V3 OFF)
    endif (DEBHELPER_COMPAT_VERSION STREQUAL "13")

    if (FPPOW GREATER 6)
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "libboost-dev, ")
    else (FPPOW GREATER 6)
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "")
    endif (FPPOW GREATER 6)

    if (ENABLE_OPENCL AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc"))
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "${CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS}opencl-headers, ocl-icd-opencl-dev, xxd")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "ocl-icd-opencl-dev")
    else (ENABLE_OPENCL AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc"))
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
    endif (ENABLE_OPENCL AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc"))

    set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
    set(CPACK_DEB_COMPONENT_INSTALL YES)
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_BINARY_DIR}/debian/rules;${CMAKE_BINARY_DIR}/debian/triggers")
    set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)
    set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)


    if (ENABLE_MAKE_CLEAN_OVERRIDE)
        configure_file (makefile.in makefile @ONLY)
    endif (ENABLE_MAKE_CLEAN_OVERRIDE)
    configure_file (debian/control.in debian/control @ONLY)
    configure_file (debian/libqrack.dirs.in debian/${CPACK_PACKAGE_NAME}.dirs @ONLY)
    configure_file (debian/libqrack.install.in debian/${CPACK_PACKAGE_NAME}.install @ONLY)
    configure_file (debian/libqrack-dev.dirs.in debian/${CPACK_PACKAGE_NAME}-dev.dirs @ONLY)
    configure_file (debian/libqrack-dev.install.in debian/${CPACK_PACKAGE_NAME}-dev.install @ONLY)
    configure_file (debian/libqrack-docs.docs.in debian/${CPACK_PACKAGE_NAME}-docs.docs @ONLY)
    file(COPY debian DESTINATION ${CMAKE_BINARY_DIR})

    file(COPY makefile.in DESTINATION ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION})
    if (NOT "${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION}")
        file(GLOB ARCHIVE_FILES ${CMAKE_BINARY_DIR}/*)
        file(COPY ${ARCHIVE_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION})
        execute_process(
            COMMAND make clean-cmake
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION}
            )
        execute_process(
            COMMAND tar -C ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION} -cJf ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}_${CMAKE_PROJECT_VERSION}.orig.tar.xz .
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION}
            )
    endif (NOT "${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION}")
else (PACK_DEBIAN)
    set(CPACK_SOURCE_GENERATOR "TGZ")
endif (PACK_DEBIAN)
include(CPack)

configure_file(include/common/config.h.in include/common/config.h @ONLY)
