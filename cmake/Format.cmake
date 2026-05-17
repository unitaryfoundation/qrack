set( FORMAT_EXCLUDE_FILES "catch.hpp" "_build")

find_program ( CLANG_FORMAT clang-format-21 )
file (GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.hpp *.cu *.cuh)

foreach (SOURCE_FILE ${ALL_SOURCE_FILES})
    foreach (EXCLUDE_FILE ${FORMAT_EXCLUDE_FILES})
		string (FIND ${SOURCE_FILE} ${EXCLUDE_FILE} EXCLUDE)
		if (NOT ${EXCLUDE} EQUAL -1)
			list(REMOVE_ITEM ALL_SOURCE_FILES ${SOURCE_FILE})
		endif ()
    endforeach ()
endforeach ()

add_custom_target (
    format
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND ${CLANG_FORMAT} -style=file -i ${ALL_SOURCE_FILES}
    )

