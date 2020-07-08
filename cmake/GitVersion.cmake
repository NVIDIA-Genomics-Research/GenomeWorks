

macro (GitVersion)
    execute_process(COMMAND
        git describe --tag --dirty
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE CLARA_PARABRICKS_GENOMEWORKS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)

endmacro ()