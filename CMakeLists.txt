cmake_minimum_required(VERSION 3.16)
project(IOOptimiationPass)

# Find the LLVM package installed on your system
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Find MLIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Add LLVM and MLIR CMake paths so we can use their custom macros (like TableGen)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# ------------------------------------------------------------------------------
# Test Suite Integration (lit)
# ------------------------------------------------------------------------------

# 1. Search the system for the 'lit' or 'llvm-lit' testing tool
find_program(LIT_TOOL NAMES llvm-lit-20 lit-20 llvm-lit lit)

if(LIT_TOOL)


    message(STATUS "Found lit test tool: ${LIT_TOOL}")


    # Find test tools on the current system
    find_program(CLANG_PATH NAMES clang-20 clang)
    find_program(CLANGPP_PATH NAMES clang++-20 clang++)
    find_program(OPT_PATH NAMES opt-20 opt)
    find_program(FILECHECK_PATH NAMES FileCheck-20 FileCheck)
    find_program(LLVM_LINK_PATH NAMES llvm-link-20 llvm-link)
     
    set(IO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(IO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(IO_SHLIB_EXT ${CMAKE_SHARED_LIBRARY_SUFFIX})

    # This creates a 'lit.site.cfg.py' in the build directory with real paths
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/llvm-tests/lit.site.cfg.py.in
	${CMAKE_CURRENT_SOURCE_DIR}/llvm-tests/lit.site.cfg.py
        @ONLY
    )


    # 2. Create a custom 'check' target
    add_custom_target(check
        # Run lit on the test/ directory, passing the -v (verbose) flag
        COMMAND ${LIT_TOOL} -v ${CMAKE_CURRENT_SOURCE_DIR}/llvm-tests
        
        # VERY IMPORTANT: Make the tests depend on the pass itself!
        # This forces CMake to recompile IOOpt.so if you changed the C++ code
        # before running the tests.
        DEPENDS IOOpt
        
        # Print a nice message when it starts
        COMMENT "Running IOOptimisationPass test suite..."
        
        # Give lit direct access to the terminal so its colored output works
        USES_TERMINAL
    )
else()
    message(WARNING "Could not find the 'lit' tool. The 'make check' target will not be created.")
    message(WARNING "You can install it via: pip install lit")
endif()

find_package(MPI)

add_subdirectory(llvm-src)
add_subdirectory(benchmarks)
add_subdirectory(mlir-src)
