# Build the runtime as a Static Library
add_library(ioopt_runtime STATIC ioopt_runtime.c)

target_compile_options(ioopt_runtime PRIVATE "-O3")

set_target_properties(ioopt_runtime PROPERTIES 
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/runtime_lib"
)
