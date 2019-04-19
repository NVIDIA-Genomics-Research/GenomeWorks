cmake_minimum_required(VERSION 3.10.2)

set_property(GLOBAL PROPERTY doxygen_src_dirs)
function(add_doxygen_source_dir)
    get_property(tmp GLOBAL PROPERTY doxygen_src_dirs)
    # avoid leading space in path
    foreach(arg ${ARGV})
        if("${tmp}" STREQUAL "")
            set(tmp "${arg}")
        else()
            set(tmp "${tmp}\;${arg}")
        endif()
    endforeach()

    set_property(GLOBAL PROPERTY doxygen_src_dirs "${tmp}")
endfunction(add_doxygen_source_dir)

function(add_docs_target DOXY_PACKAGE_NAME DOXY_PACKAGE_VERSION)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_PROJECT_NAME ${DOXY_PACKAGE_NAME})
        set(DOXYGEN_PROJECT_NUMBER ${DOXY_PACKAGE_VERSION})

        get_property(final_doxygen_src_dirs GLOBAL PROPERTY doxygen_src_dirs)
        doxygen_add_docs(docs ALL ${final_doxygen_src_dirs})
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/ DESTINATION docs)
    else()
        message(STATUS "Doxygen not found. Doc gen disabled.")
    endif()

endfunction(add_docs_target)
