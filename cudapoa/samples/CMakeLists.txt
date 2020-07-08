

project(sample_cudapoa)

get_property(cudapoa_data_include_dir GLOBAL PROPERTY cudapoa_data_include_dir)
include_directories(${cudapoa_data_include_dir})

add_executable(${PROJECT_NAME}
               sample_cudapoa.cpp
               )

target_link_libraries(${PROJECT_NAME}
                      cudapoa
                      )

install(TARGETS ${PROJECT_NAME}
            DESTINATION samples)
