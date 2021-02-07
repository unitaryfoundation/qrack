option (ENABLE_PURE32 "Use only 32-bit types or smaller" OFF)
option (ENABLE_VC4CL "Build a library version that's safe for the VC4CL compiler (for the Raspberry Pi 3)" OFF)
option (ENABLE_UINT32 "Use 32-bit (instead of 64-bit) unsigned integer types for coherent addressable qubit masks" OFF)

if (ENABLE_VC4CL)
    set(ENABLE_PURE32 ON)
    target_compile_definitions(qrack PUBLIC ENABLE_VC4CL=1)
endif(ENABLE_VC4CL)

if (ENABLE_PURE32 OR (QBCAPPOW EQUAL 5))
    set(ENABLE_COMPLEX_X2 OFF)
    set(ENABLE_COMPLEX8 ON)
    set(ENABLE_INT32 ON)
    set(QBCAPPOW "5")
endif (ENABLE_PURE32 OR (QBCAPPOW EQUAL 5))

if (ENABLE_UINT32)
    target_compile_definitions (qrack PUBLIC ENABLE_PURE32=1)
endif (ENABLE_UINT32)
