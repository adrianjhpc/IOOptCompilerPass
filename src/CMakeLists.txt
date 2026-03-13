# Include LLVM's headers and definitions
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

# Our pass is a loadable module (shared library)
add_library(IOOpt MODULE IOOptimisationPass.cpp)

# Require C++17 (Standard for modern LLVM) and Position Independent Code
set_target_properties(IOOpt PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)

# In Windows/Mac, undefined symbols are not allowed in shared libraries by default.
# LLVM passes often have them because the symbols are resolved when `opt` loads the pass.
if(APPLE)
  set_target_properties(IOOpt PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()
