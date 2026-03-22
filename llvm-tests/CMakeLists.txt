# ==============================================================================
# Hourglass Architecture Integration Tests (LLVM LTO)
# ==============================================================================

# Find the LLVM tools needed for the LTO simulation
find_program(LLVM_LINK_EXE llvm-link HINTS ${LLVM_TOOLS_BINARY_DIR})
find_program(OPT_EXE opt HINTS ${LLVM_TOOLS_BINARY_DIR})

if(LLVM_LINK_EXE AND OPT_EXE)
    add_test(
        NAME HourglassLTOTest
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_fullstack_test.sh 
                ${CMAKE_C_COMPILER}        
                ${LLVM_LINK_EXE}      
                ${OPT_EXE}         
                $<TARGET_FILE:IOOpt> 
    )
else()
    message(WARNING "llvm-link or opt not found! Skipping Hourglass LTO tests.")
endif()
