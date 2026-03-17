# Find the Clang executable inside your custom LLVM installation
find_program(CLANG_EXE clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

if(NOT CLANG_EXE)
    message(WARNING "Could not find 'clang' in the LLVM installation. Tests will fail.")
endif()

# Define the test
add_test(NAME AutoZeroCopyTest
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_zerocopy_test.sh
            ${CLANG_EXE}                     # Argument 1: Path to Clang
            $<TARGET_FILE:io-opt>            # Argument 2: Path to your compiled io-opt binary
            ${CMAKE_CURRENT_SOURCE_DIR}/zerocopy_test.c # Argument 3: Path to the C file
)

set_tests_properties(AutoZeroCopyTest PROPERTIES TIMEOUT 10)
