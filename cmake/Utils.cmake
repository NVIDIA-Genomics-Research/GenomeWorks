

function(validate_boolean CMAKE_OPTION)
    if ((NOT ${CMAKE_OPTION} STREQUAL "ON") AND (NOT ${CMAKE_OPTION} STREQUAL "OFF"))
        message(FATAL_ERROR "${CMAKE_OPTION}  can only be set to ON/OFF")
    endif()
endfunction(validate_boolean)
