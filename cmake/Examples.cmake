add_executable (quantum_associative_memory
    examples/quantum_associative_memory.cpp
    )
set_target_properties(quantum_associative_memory PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (quantum_associative_memory ${QRACK_LIBS})

add_executable (teleport
    examples/teleport.cpp
    )
set_target_properties(teleport PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (teleport ${QRACK_LIBS})

add_executable (qneuron_classification
    examples/qneuron_classification.cpp
    )
set_target_properties(qneuron_classification PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (qneuron_classification ${QRACK_LIBS})

add_executable (cosmology
    examples/cosmology.cpp
    )
set_target_properties(cosmology PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (cosmology ${QRACK_LIBS})

if ((FPPOW EQUAL 5) OR (FPPOW EQUAL 6))
    if(POLICY CMP0074)
        cmake_policy(SET CMP0074 NEW)
        find_package(Eigen3 CONFIG)
    else(POLICY CMP0074)
        find_package(Eigen3)
    endif(POLICY CMP0074)
    if (Eigen3_FOUND)
        add_executable (separability
            examples/separability.cpp
            )
        set_target_properties(separability PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
        target_link_libraries (separability ${QRACK_LIBS} Eigen3::Eigen)
        
        add_executable (qbdd_separability
            examples/qbdd_separability.cpp
            )
        set_target_properties(qbdd_separability PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
        target_link_libraries (qbdd_separability ${QRACK_LIBS} Eigen3::Eigen)

        add_executable (qunit_separability
            examples/qunit_separability.cpp
            )
        set_target_properties(qunit_separability PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
        target_link_libraries (qunit_separability ${QRACK_LIBS} Eigen3::Eigen)
    endif (Eigen3_FOUND)
endif ((FPPOW EQUAL 5) OR (FPPOW EQUAL 6))

if (ENABLE_ALU)
add_executable (grovers
    examples/grovers.cpp
    )
set_target_properties(grovers PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (grovers ${QRACK_LIBS})

add_executable (grovers_lookup
    examples/grovers_lookup.cpp
    )
set_target_properties(grovers_lookup PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (grovers_lookup ${QRACK_LIBS})

add_executable (ordered_list_search
    examples/ordered_list_search.cpp
    )
set_target_properties(ordered_list_search PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (ordered_list_search ${QRACK_LIBS})

add_executable (shors_factoring
    examples/shors_factoring.cpp
    )
set_target_properties(shors_factoring PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (shors_factoring ${QRACK_LIBS})

add_executable (pearson32
    examples/pearson32.cpp
    )
set_target_properties(pearson32 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (pearson32 ${QRACK_LIBS})

add_executable (quantum_perceptron
    examples/quantum_perceptron.cpp
    )
set_target_properties(quantum_perceptron PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (quantum_perceptron ${QRACK_LIBS})
endif (ENABLE_ALU)

configure_file(examples/data/powers_of_2.csv examples/data/powers_of_2.csv COPYONLY)

set(EXAMPLE_COMPILE_OPTS ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (quantum_associative_memory PUBLIC ${EXAMPLE_COMPILE_OPTS})
target_compile_options (teleport PUBLIC ${EXAMPLE_COMPILE_OPTS})
target_compile_options (qneuron_classification PUBLIC ${EXAMPLE_COMPILE_OPTS})
if (ENABLE_ALU)
    target_compile_options (grovers PUBLIC ${EXAMPLE_COMPILE_OPTS})
    target_compile_options (grovers_lookup PUBLIC ${EXAMPLE_COMPILE_OPTS})
    target_compile_options (ordered_list_search PUBLIC ${EXAMPLE_COMPILE_OPTS})
    target_compile_options (shors_factoring PUBLIC ${EXAMPLE_COMPILE_OPTS})
    target_compile_options (pearson32 PUBLIC ${EXAMPLE_COMPILE_OPTS})
    target_compile_options (quantum_perceptron PUBLIC ${EXAMPLE_COMPILE_OPTS})
endif (ENABLE_ALU)
