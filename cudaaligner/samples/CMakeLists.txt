

project(sample_cudaaligner)

add_executable(${PROJECT_NAME}
               sample_cudaaligner.cpp
               )

target_link_libraries(${PROJECT_NAME}
                      cudaaligner
                      )

install(TARGETS ${PROJECT_NAME}
            DESTINATION samples)
