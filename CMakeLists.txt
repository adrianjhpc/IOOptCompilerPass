cmake_minimum_required(VERSION 3.16)
project(IOOptimiationPass)

# Find the LLVM package installed on your system
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# ------------------------------------------------------------------------------
# Test Suite Integration (lit)
# ------------------------------------------------------------------------------

# 1. Search the system for the 'lit' or 'llvm-lit' testing tool
find_program(LIT_TOOL NAMES llvm-lit-20 lit-20 llvm-lit lit)

if(LIT_TOOL)
    message(STATUS "Found lit test tool: ${LIT_TOOL}")

    # 2. Create a custom 'check' target
    add_custom_target(check
        # Run lit on the test/ directory, passing the -v (verbose) flag
        COMMAND ${LIT_TOOL} -v ${CMAKE_CURRENT_SOURCE_DIR}/tests
        
        # VERY IMPORTANT: Make the tests depend on the pass itself!
        # This forces CMake to recompile IOOpt.so if you changed the C++ code
        # before running the tests.
        DEPENDS IOOpt
        
        # Print a nice message when it starts
        COMMENT "Running IOOptimizationPass test suite..."
        
        # Give lit direct access to the terminal so its colored output works
        USES_TERMINAL
    )
else()
    message(WARNING "Could not find the 'lit' tool. The 'make check' target will not be created.")
    message(WARNING "You can install it via: pip install lit")
endif()

add_subdirectory(src)
add_subdirectory(benchmarks)
